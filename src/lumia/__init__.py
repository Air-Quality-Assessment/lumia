#!/usr/bin/env python

# Base objects (default versions)
from .observations.observations import Observations
from .observations.observations_14c import Observations14C
from .data.xr import Data
from .prior.prior import PriorConstraints
from .mapping.multitracer import Mapping
from .optimizer.scipy_optimizer import Optimizer
from .models.footprints.transport import Transport
from .models.footprints.transport_14c import Transport as Transport14C

# Utilities
from .utils.dconf import read_config, write_config

from types import SimpleNamespace
settings = SimpleNamespace(read = read_config, write = write_config)

from .utils.logging import setup_logging
setup_logging()

