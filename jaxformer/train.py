"""Training loop: Optax optimizer, jitted sharded step, Orbax checkpointing.

The NNX model is split once into a static ``graphdef`` and a pytree of parameters, and
everything after that is plain functional JAX. That keeps the jitted step transparent —
there is no framework magic between the loss and the optimizer — and it means the same
step function runs unchanged on 1 CPU device or 8 TPU chips.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Any, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jaxformer import sharding as shd
from jaxformer.config import ModelConfig, TrainConfig
from jaxformer.model import Transformer

Params = Any  # nnx.State — a pytree with jax.Array leaves


class TrainState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    step: jax.Array


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _decay_mask(params: Params) -> Any:
    """Decay matmul weights; leave 1-D parameters (RMSNorm scales) alone.

    Weight-decaying a normalization scale fights the normalizer and costs a little
    loss for nothing. Standard practice since GPT-2.
    """
    return jax.tree.map(lambda p: p.ndim >= 2, params)


def build_optimizer(cfg: TrainConfig, params: Params) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.total_steps,
        end_value=cfg.min_learning_rate,
    )
    return optax.chain(
        # Clip before the optimizer, so Adam's second moment never sees a spike.
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.b1,
            b2=cfg.b2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            mask=_decay_mask(params),
        ),
    )


def lr_at(cfg: TrainConfig, step: int) -> float:
    """The schedule value at a step, for logging."""
    return float(
        optax.warmup_cosine_decay_schedule(
            0.0, cfg.learning_rate, cfg.warmup_steps, cfg.total_steps, cfg.min_learning_rate
        )(step)
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def loss_fn(params: Params, graphdef, batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Next-token cross entropy. ``batch`` is ``(B, T+1)`` int32.

    Returns (mean loss in nats, mean logit magnitude) — the second is a cheap
    divergence canary; logits drifting upward is the first visible sign of instability.
    """
    model = nnx.merge(graphdef, params)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean(), jnp.mean(jnp.abs(logits))


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def make_train_step(graphdef, tx: optax.GradientTransformation, accum_steps: int):
    """Build the jitted training step.

    ``graphdef`` and ``tx`` are closed over rather than passed as arguments because
    both are static; passing them would force a retrace on every call.
    """

    def micro_grads(params, micro_batch):
        (loss, logit_mag), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, graphdef, micro_batch
        )
        return loss, logit_mag, grads

    @functools.partial(jax.jit, donate_argnums=(0,))
    def step(state: TrainState, batch: jax.Array) -> tuple[TrainState, dict]:
        if accum_steps == 1:
            loss, logit_mag, grads = micro_grads(state.params, batch)
        else:
            # Scan over micro-batches instead of unrolling: one copy of the backward
            # pass in the HLO regardless of accumulation depth, so compile time and
            # peak memory stay flat as the effective batch grows.
            B = batch.shape[0]
            micro = batch.reshape(accum_steps, B // accum_steps, *batch.shape[1:])

            def body(carry, mb):
                loss_sum, mag_sum, grad_sum = carry
                loss, mag, grads = micro_grads(state.params, mb)
                grad_sum = jax.tree.map(jnp.add, grad_sum, grads)
                return (loss_sum + loss, mag_sum + mag, grad_sum), None

            zeros = jax.tree.map(jnp.zeros_like, state.params)
            (loss_sum, mag_sum, grads), _ = jax.lax.scan(
                body, (jnp.zeros(()), jnp.zeros(()), zeros), micro
            )
            scale = 1.0 / accum_steps
            loss, logit_mag = loss_sum * scale, mag_sum * scale
            grads = jax.tree.map(lambda g: g * scale, grads)

        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics = {
            "loss": loss,
            "logit_mag": logit_mag,
            "grad_norm": optax.tree.norm(grads),
        }
        return TrainState(params, opt_state, state.step + 1), metrics

    return step


def make_eval_step(graphdef):
    @jax.jit
    def evaluate(params, batch):
        loss, _ = loss_fn(params, graphdef, batch)
        return loss

    return evaluate


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def init_train_state(
    model_cfg: ModelConfig, train_cfg: TrainConfig, mesh, compute_dtype=jnp.bfloat16
) -> tuple[Any, TrainState, optax.GradientTransformation]:
    """Build the model, split it, and place parameters replicated across the mesh."""
    model = Transformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed), compute_dtype=compute_dtype)
    graphdef, params = nnx.split(model)
    tx = build_optimizer(train_cfg, params)

    params = shd.put_replicated(params, mesh)
    opt_state = shd.put_replicated(jax.jit(tx.init)(params), mesh)
    state = TrainState(params, opt_state, jnp.asarray(0, jnp.int32))
    return graphdef, state, tx


def param_count(params: Params) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(params))


def flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """Training FLOPs per token, forward + backward (PaLM appendix-B convention).

    ``6 * N`` for the weight matmuls (2 for forward, 4 for backward), plus the
    attention score and value matmuls, which are not weight FLOPs and grow with
    sequence length.
    """
    non_embed = cfg.param_count()["layers"]
    attn = 12 * cfg.n_layers * cfg.d_model * seq_len
    return 6 * non_embed + attn


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepLog:
    step: int
    loss: float
    grad_norm: float
    lr: float
    tokens_per_sec: float
    mfu: float | None


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_batches: Iterator[jax.Array],
    val_batches=None,
    *,
    mesh=None,
    checkpoint_dir: str | None = None,
    peak_flops_per_device: float | None = None,
    compute_dtype=jnp.bfloat16,
    on_log=None,
    on_eval=None,
):
    """Run the training loop. ``train_batches`` yields ``(B, T+1)`` int32 arrays.

    ``on_log(StepLog)`` fires at each train-loss log; ``on_eval(step, val_loss)`` fires
    at each validation, so callers can collect both series without parsing stdout.
    """
    mesh = mesh or shd.make_mesh()
    graphdef, state, tx = init_train_state(model_cfg, train_cfg, mesh, compute_dtype)
    step_fn = make_train_step(graphdef, tx, train_cfg.grad_accum_steps)
    eval_fn = make_eval_step(graphdef)

    n_params = param_count(state.params)
    fpt = flops_per_token(model_cfg, model_cfg.max_seq_len)
    tokens_per_step = train_cfg.tokens_per_step(model_cfg.max_seq_len)
    print(f"mesh:   {shd.describe(mesh)}")
    print(f"params: {n_params:,}")
    print(f"tokens/step: {tokens_per_step:,}  total: {tokens_per_step * train_cfg.total_steps:,}")

    ckptr = None
    if checkpoint_dir:
        import orbax.checkpoint as ocp

        ckptr = ocp.StandardCheckpointer()

    logs: list[StepLog] = []
    t0 = time.perf_counter()
    window_start, window_steps = t0, 0

    for step in range(1, train_cfg.total_steps + 1):
        batch = shd.put_batch(next(train_batches), mesh)
        state, metrics = step_fn(state, batch)
        window_steps += 1

        if step % train_cfg.log_every == 0:
            # Metrics are device arrays; blocking here is also what makes the timing
            # window meaningful rather than measuring async dispatch.
            metrics = jax.device_get(metrics)
            now = time.perf_counter()
            tps = window_steps * tokens_per_step / (now - window_start)
            mfu = None
            if peak_flops_per_device:
                mfu = tps * fpt / (peak_flops_per_device * mesh.size)
            log = StepLog(
                step, float(metrics["loss"]), float(metrics["grad_norm"]),
                lr_at(train_cfg, step), tps, mfu,
            )
            logs.append(log)
            msg = (
                f"step {step:>6}  loss {log.loss:.4f}  gnorm {log.grad_norm:.3f}  "
                f"lr {log.lr:.2e}  {tps / 1e3:.1f}k tok/s"
            )
            if mfu is not None:
                msg += f"  mfu {mfu * 100:.1f}%"
            print(msg)
            if on_log:
                on_log(log)
            window_start, window_steps = now, 0

        if val_batches is not None and step % train_cfg.eval_every == 0:
            val = jnp.mean(
                jnp.stack([
                    eval_fn(state.params, shd.put_batch(next(val_batches), mesh))
                    for _ in range(train_cfg.eval_steps)
                ])
            )
            print(f"step {step:>6}  val_loss {float(val):.4f}")
            if on_eval:
                on_eval(step, float(val))
            window_start, window_steps = time.perf_counter(), 0

        if ckptr and checkpoint_dir and step % train_cfg.checkpoint_every == 0:
            save_checkpoint(ckptr, checkpoint_dir, step, state.params)

    if ckptr and checkpoint_dir:
        save_checkpoint(ckptr, checkpoint_dir, train_cfg.total_steps, state.params)
        # Orbax writes asynchronously. Without this the final checkpoint is still a
        # `.orbax-checkpoint-tmp` directory when the process exits — which on Kaggle
        # means losing the run you just spent nine hours of quota on.
        ckptr.wait_until_finished()

    print(f"done in {time.perf_counter() - t0:.1f}s")
    return graphdef, state, logs


def save_checkpoint(ckptr, checkpoint_dir: str, step: int, params: Params) -> str:
    import os

    path = os.path.abspath(os.path.join(checkpoint_dir, f"step_{step}"))
    ckptr.save(path, params, force=True)
    return path


def load_checkpoint(path: str, template: Params) -> Params:
    """Restore parameters. ``template`` supplies the expected tree structure/shapes."""
    import os

    import orbax.checkpoint as ocp

    return ocp.StandardCheckpointer().restore(os.path.abspath(path), target=template)
