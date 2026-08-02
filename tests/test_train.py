"""Sharding and training-loop tests, run against 8 simulated CPU devices.

The load-bearing test here is ``test_sharded_step_matches_single_device``. Data
parallelism is easy to get subtly wrong — a batch split along the wrong axis, or an
optimizer state that drifts per-device — and the failure mode is not a crash but a
model that trains slightly worse than it should. Pinning 8-device and 1-device results
to each other on a laptop is far cheaper than discovering the discrepancy on Kaggle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxformer import sharding as shd
from jaxformer.config import TrainConfig, tiny
from jaxformer.train import (
    build_optimizer,
    flops_per_token,
    init_train_state,
    make_train_step,
    param_count,
)


def synthetic_batches(vocab: int, batch: int, seq: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    while True:
        yield jnp.asarray(rng.integers(0, vocab, size=(batch, seq + 1)), jnp.int32)


@pytest.fixture(scope="module")
def train_cfg() -> TrainConfig:
    return TrainConfig(
        batch_size=8, total_steps=8, warmup_steps=2, eval_every=1000,
        checkpoint_every=1000, log_every=1000,
    )


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


def test_simulated_device_count():
    assert jax.device_count() == 8, "conftest.py should fake 8 CPU devices"


def test_mesh_covers_all_devices():
    mesh = shd.make_mesh()
    assert mesh.size == jax.device_count()
    assert mesh.axis_names == (shd.DATA_AXIS,)


def test_put_batch_rejects_indivisible_batch():
    mesh = shd.make_mesh()
    with pytest.raises(ValueError, match="not divisible"):
        shd.put_batch(jnp.zeros((mesh.size + 1, 4), jnp.int32), mesh)


def test_put_batch_splits_along_axis_zero():
    mesh = shd.make_mesh()
    x = shd.put_batch(jnp.zeros((16, 5), jnp.int32), mesh)
    # 16 rows over 8 devices -> 2 rows each.
    assert x.sharding.shard_shape(x.shape) == (16 // mesh.size, 5)


def test_params_are_replicated_not_sharded(train_cfg):
    mesh = shd.make_mesh()
    _, state, _ = init_train_state(tiny(), train_cfg, mesh, compute_dtype=jnp.float32)
    for leaf in jax.tree.leaves(state.params):
        assert leaf.sharding.shard_shape(leaf.shape) == leaf.shape


# ---------------------------------------------------------------------------
# The equivalence that matters
# ---------------------------------------------------------------------------


def test_sharded_step_matches_single_device(train_cfg):
    """Eight devices and one device must produce the same parameters."""
    cfg = tiny()
    batch = next(synthetic_batches(cfg.vocab_size, 8, 16))

    def run(devices):
        mesh = shd.make_mesh(devices)
        graphdef, state, tx = init_train_state(
            cfg, train_cfg, mesh, compute_dtype=jnp.float32
        )
        step = make_train_step(graphdef, tx, accum_steps=1)
        for _ in range(3):
            state, metrics = step(state, shd.put_batch(batch, mesh))
        return jax.device_get(state.params), jax.device_get(metrics)

    multi_params, multi_metrics = run(jax.devices())
    single_params, single_metrics = run(jax.devices()[:1])

    assert float(multi_metrics["loss"]) == pytest.approx(
        float(single_metrics["loss"]), rel=1e-5
    )
    # atol=1e-5 rather than exact equality: summing gradients across 8 shards uses a
    # different reduction order than summing them on one device, and float32 addition
    # is not associative. Empirically this shows up as a handful of elements out of
    # ~45k differing in the 6th decimal. A tighter bound here would be a test that
    # fails for arithmetic reasons rather than correctness ones.
    for a, b in zip(jax.tree.leaves(multi_params), jax.tree.leaves(single_params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5)


def test_grad_accumulation_matches_single_pass(train_cfg):
    """Accumulating two micro-batches equals one pass over the full batch."""
    cfg = tiny()
    batch = next(synthetic_batches(cfg.vocab_size, 8, 16))
    mesh = shd.make_mesh(jax.devices()[:1])

    def run(accum):
        graphdef, state, tx = init_train_state(
            cfg, train_cfg, mesh, compute_dtype=jnp.float32
        )
        step = make_train_step(graphdef, tx, accum_steps=accum)
        state, metrics = step(state, shd.put_batch(batch, mesh))
        return jax.device_get(state.params), float(metrics["loss"])

    p1, loss1 = run(1)
    p2, loss2 = run(2)

    assert loss1 == pytest.approx(loss2, rel=1e-5)
    for a, b in zip(jax.tree.leaves(p1), jax.tree.leaves(p2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Optimizer behaviour
# ---------------------------------------------------------------------------


def test_weight_decay_skips_one_dim_params(train_cfg):
    """RMSNorm scales must not be decayed."""
    from jaxformer.train import _decay_mask

    mesh = shd.make_mesh(jax.devices()[:1])
    _, state, _ = init_train_state(tiny(), train_cfg, mesh, compute_dtype=jnp.float32)
    mask = _decay_mask(state.params)
    flags = jax.tree.leaves(mask)
    dims = [p.ndim for p in jax.tree.leaves(state.params)]
    assert any(d == 1 for d in dims), "expected some 1-D params (norm scales)"
    for flag, ndim in zip(flags, dims):
        assert flag == (ndim >= 2)


def test_learning_rate_warms_up_then_decays():
    cfg = TrainConfig(warmup_steps=100, total_steps=1000)
    from jaxformer.train import lr_at

    assert lr_at(cfg, 0) < lr_at(cfg, 50) < lr_at(cfg, 100)
    assert lr_at(cfg, 100) == pytest.approx(cfg.learning_rate, rel=1e-6)
    assert lr_at(cfg, 1000) < lr_at(cfg, 500) < lr_at(cfg, 100)
    assert lr_at(cfg, 1000) == pytest.approx(cfg.min_learning_rate, rel=1e-3)


def test_loss_decreases_on_a_memorizable_batch(train_cfg):
    """Sanity: the same batch repeatedly should be memorized quickly."""
    cfg = tiny()
    mesh = shd.make_mesh()
    batch = next(synthetic_batches(cfg.vocab_size, 8, 16))
    # Own config: the shared fixture sets total_steps=8, so the cosine schedule would
    # bottom out at min_lr a quarter of the way through this loop and the model would
    # barely move for the rest of it.
    cfg_t = TrainConfig(
        batch_size=8, total_steps=30, warmup_steps=2, learning_rate=3e-3,
        eval_every=1000, checkpoint_every=1000, log_every=1000,
    )
    graphdef, state, tx = init_train_state(cfg, cfg_t, mesh, compute_dtype=jnp.float32)
    step = make_train_step(graphdef, tx, accum_steps=1)

    losses = []
    for _ in range(30):
        state, metrics = step(state, shd.put_batch(batch, mesh))
        losses.append(float(metrics["loss"]))

    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] * 0.7, f"loss barely moved: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_initial_loss_is_near_uniform(train_cfg):
    """At init the model should predict ~uniform, i.e. loss ~= ln(vocab_size).

    A meaningfully different starting loss means the initialization scale is wrong.
    """
    cfg = tiny()
    mesh = shd.make_mesh()
    graphdef, state, tx = init_train_state(cfg, train_cfg, mesh, compute_dtype=jnp.float32)
    step = make_train_step(graphdef, tx, accum_steps=1)
    batch = next(synthetic_batches(cfg.vocab_size, 8, 16))
    _, metrics = step(state, shd.put_batch(batch, mesh))
    assert float(metrics["loss"]) == pytest.approx(np.log(cfg.vocab_size), rel=0.05)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_exactly(tmp_path, train_cfg):
    """Save then restore must be bit-exact, and must leave no unfinished temp dir.

    Orbax writes asynchronously; a checkpoint that has not been waited on is still a
    ``.orbax-checkpoint-tmp`` directory. On Kaggle that is the difference between
    keeping and losing a nine-hour run.
    """
    import orbax.checkpoint as ocp

    from jaxformer.train import load_checkpoint, save_checkpoint

    mesh = shd.make_mesh()
    _, state, _ = init_train_state(tiny(), train_cfg, mesh, compute_dtype=jnp.float32)

    ckptr = ocp.StandardCheckpointer()
    path = save_checkpoint(ckptr, str(tmp_path), 1, state.params)
    ckptr.wait_until_finished()

    assert not [p for p in tmp_path.iterdir() if p.name.endswith("tmp")]
    restored = load_checkpoint(path, state.params)
    for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_flops_per_token_grows_with_sequence_length():
    cfg = tiny()
    assert flops_per_token(cfg, 2048) > flops_per_token(cfg, 512)


def test_param_count_agrees_with_config(train_cfg):
    cfg = tiny()
    mesh = shd.make_mesh(jax.devices()[:1])
    _, state, _ = init_train_state(cfg, train_cfg, mesh, compute_dtype=jnp.float32)
    assert param_count(state.params) == cfg.param_count()["total"]
