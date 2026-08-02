"""Test-session setup.

XLA reads device-count flags once, at backend initialization, so this must run
before anything imports jax. Putting it in conftest.py rather than pyproject.toml
avoids a dependency on the pytest-env plugin.

Eight fake CPU devices are what let the whole sharding path in ``jaxformer.sharding``
be developed and tested on a laptop, before any Kaggle TPU quota is spent on it.
"""

import os

_XLA_FLAGS = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _XLA_FLAGS:
    os.environ["XLA_FLAGS"] = f"{_XLA_FLAGS} --xla_force_host_platform_device_count=8".strip()

# Keep the tests off any accelerator so results are identical everywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
