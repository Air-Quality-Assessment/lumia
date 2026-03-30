#!/usr/bin/env python

from dataclasses import dataclass, field
from pandas import DataFrame
from numpy.typing import NDArray
from .protocols import Departures, Model, Mapping, Prior, Observations
from omegaconf import DictConfig
from functools import partial
from pathlib import Path
from numpy import zeros, inner
from loguru import logger
from ..minimizers.congrad import Minimizer as Congrad
from .preconditioning import xc_to_x, g_to_gc
from lumia.utils.dconf import prefix
from typing import Dict

from .optimizer import CostFunction, Settings
from .optimizer import Var4D as BaseVar4D

@dataclass(kw_only=True)
class Var4D(BaseVar4D) :
    prior : Prior                           # Contains the prior uncertainty (B) as sigmas + correlation matrices
    model : Model                           # Calculates y = H(X) and x* = H*(dy)
    mapping : Mapping                       # Contains the mapping between state vector and model params
    observations : Observations             # Contains the obs, obs uncertainties
    settings : Settings | DictConfig | Dict # settings requires by the optimizer class (i.e. this class)
    iteration : int = 0
    _vectors : DataFrame | None = None
    
    def forward_step(self, state_preco: NDArray) -> Departures:
        state = self.xc_to_x(state_preco)
        model_data = self.mapping.vec_to_struct(state)
        return self.model.calc_departures(model_data, step=self.step)

    def adjoint_step(self, obs_departures : Departures) -> NDArray:
        model_data_adj = self.model.calc_departures_adj(obs_departures.mismatch / obs_departures.sigma ** 2)
        state_adj = self.mapping.vec_to_struct_adj(model_data_adj)
        return self.g_to_gc(state_adj)


def adjoint_test(opt: Var4D):
    from numpy import random
    from copy import deepcopy

    opt.model.observations.loc[:, 'err'] = 2.

    xref = opt.prior.state_preco.copy()
    yref = opt.forward_step(xref)

    x1 = random.randn(opt.prior.size)
    y1 = opt.forward_step(x1)

    y2 = deepcopy(y1)
    y2.mismatch = random.randn(len(y2.mismatch))
    x2 = opt.adjoint_step(y2)

    dy = y1.mismatch - yref.mismatch
    w = y2.mismatch / y2.sigma**2

    lhs = dy @ w
    rhs = x2 @ x1

    logger.info(
        f"Adjoint test: lhs={lhs:.12e}, rhs={rhs:.12e}, "
        f"relerr={abs(lhs-rhs)/max(1.0, abs(lhs), abs(rhs)):.3e}"
    )

def gradient_test(opt: Var4D):
    from numpy import random

    opt.model.observations.loc[:, 'err'] = 2.

    x0 = random.randn(opt.prior.size)
    d = random.randn(opt.prior.size)

    y0 = opt.forward_step(x0)
    p0 = x0 - opt.prior.state_preco
    g0 = opt.adjoint_step(y0) + p0

    for alpha in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        xp = x0 + alpha * d
        xm = x0 - alpha * d

        yp = opt.forward_step(xp)
        ym = opt.forward_step(xm)

        Jp = CostFunction(prior_departures=xp - opt.prior.state_preco, obs_departures=yp)
        Jm = CostFunction(prior_departures=xm - opt.prior.state_preco, obs_departures=ym)

        fd = (Jp.value - Jm.value) / (2 * alpha)
        ad = g0 @ d

        logger.info(
            f"Gradient test alpha={alpha:.0e}: "
            f"fd={fd:.12e}, ad={ad:.12e}, "
            f"relerr={abs(fd-ad)/max(1.0, abs(fd), abs(ad)):.3e}"
        )

def gradient_test_prior_only(opt: Var4D):
    from numpy import random

    x0 = random.randn(opt.prior.size)
    d = random.randn(opt.prior.size)

    g_prior = x0 - opt.prior.state_preco

    for alpha in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        xp = x0 + alpha * d
        xm = x0 - alpha * d

        Jp = 0.5 * ((xp - opt.prior.state_preco) @ (xp - opt.prior.state_preco))
        Jm = 0.5 * ((xm - opt.prior.state_preco) @ (xm - opt.prior.state_preco))

        fd = (Jp - Jm) / (2 * alpha)
        ad = g_prior @ d

        logger.info(
            f"Prior-only alpha={alpha:.0e}: "
            f"fd={fd:.12e}, ad={ad:.12e}, "
            f"relerr={abs(fd-ad)/max(1.0, abs(fd), abs(ad)):.3e}"
        )

def gradient_test_obs_only(opt: Var4D):
    from numpy import random

    opt.model.observations.loc[:, 'err'] = 2.

    x0 = random.randn(opt.prior.size)
    d = random.randn(opt.prior.size)

    y0 = opt.forward_step(x0)
    g_obs = opt.adjoint_step(y0)

    for alpha in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        xp = x0 + alpha * d
        xm = x0 - alpha * d

        yp = opt.forward_step(xp)
        ym = opt.forward_step(xm)

        Jp = 0.5 * ((yp.mismatch / yp.sigma) @ (yp.mismatch / yp.sigma))
        Jm = 0.5 * ((ym.mismatch / ym.sigma) @ (ym.mismatch / ym.sigma))

        fd = (Jp - Jm) / (2 * alpha)
        ad = g_obs @ d

        logger.info(
            f"Obs-only alpha={alpha:.0e}: "
            f"fd={fd:.12e}, ad={ad:.12e}, "
            f"relerr={abs(fd-ad)/max(1.0, abs(fd), abs(ad)):.3e}"
        )