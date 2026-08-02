"""Device mesh and sharding helpers.

Pure data parallelism: parameters and optimizer state are replicated on every device,
the global batch is split along its leading axis. At 55M parameters the full optimizer
state is ~660MB in fp32, which fits comfortably in a single v5e chip's 16GB HBM, so
sharding the parameters themselves would buy nothing and cost communication.

The same code path runs on 1 device or 8. That is the point: everything here is
developed and tested against eight *simulated* CPU devices on a laptop
(``--xla_force_host_platform_device_count=8``) before any TPU quota is spent.
"""

from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

DATA_AXIS = "data"


def make_mesh(devices=None, axis_name: str = DATA_AXIS) -> Mesh:
    """One-dimensional mesh over every available device.

    Pinned to ``AxisType.Auto`` rather than taking the default. As of JAX 0.10 the
    default is ``Explicit`` ("sharding in types"), which carries shardings in array
    avals and requires every ``jit`` call to run inside a ``jax.set_mesh`` context.
    Explicit mode catches sharding mistakes at trace time and is where JAX is heading,
    but it is recent enough that the Kaggle TPU image may predate it. Auto mode leans
    on GSPMD propagation instead and behaves identically across versions, which is
    worth more here than trace-time checking — the equivalence test in
    ``tests/test_train.py`` already pins 8-device results to 1-device results.
    """
    devices = jax.devices() if devices is None else devices
    return jax.make_mesh(
        (len(devices),), (axis_name,), devices=devices, axis_types=(jax.sharding.AxisType.Auto,)
    )


def replicated(mesh: Mesh) -> NamedSharding:
    """Sharding for parameters and optimizer state — one full copy per device."""
    return NamedSharding(mesh, P())


def batch_sharded(mesh: Mesh, axis_name: str = DATA_AXIS) -> NamedSharding:
    """Sharding for a batch — split along axis 0, replicated along the rest."""
    return NamedSharding(mesh, P(axis_name))


def put_replicated(tree, mesh: Mesh):
    return jax.device_put(tree, replicated(mesh))


def put_batch(tree, mesh: Mesh):
    """Place a global batch, splitting it across devices.

    Raises rather than silently padding if the batch does not divide evenly: a ragged
    final shard would make throughput numbers quietly wrong.
    """
    n = mesh.size
    for leaf in jax.tree.leaves(tree):
        if leaf.shape[0] % n:
            raise ValueError(
                f"batch dim {leaf.shape[0]} is not divisible by device count {n}"
            )
    return jax.device_put(tree, batch_sharded(mesh))


def describe(mesh: Mesh) -> str:
    kinds = {d.device_kind for d in mesh.devices.flat}
    return (
        f"{mesh.size}x {'/'.join(sorted(kinds))} "
        f"({jax.devices()[0].platform}), mesh axes {dict(zip(mesh.axis_names, mesh.devices.shape))}"
    )
