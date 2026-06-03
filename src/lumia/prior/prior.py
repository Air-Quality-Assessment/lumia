#!/usr/bin/env python

from dataclasses import dataclass
from typing import Dict, List
from pandas import DataFrame, concat
from pandas.tseries.frequencies import to_offset
from .protocols import Mapping
from loguru import logger
from .uncertainties import calc_temporal_correlation, calc_horizontal_correlation, calc_total_uncertainty
from numpy import zeros
from lumia.optimizer.categories import Category
from pathlib import Path
from lumia.utils import debug, tracers
from omegaconf import DictConfig
import xarray as xr
from lumia.utils.units import units_registry as ureg


@dataclass
class PriorConstraints:
    temporal_correlations: Dict
    horizontal_correlations: Dict
    sigmas: Dict
    vectors: DataFrame

    def __post_init__(self):
        self.state_preco = zeros(self.size)

    @property
    def coordinates(self) -> DataFrame:
        return self.vectors.loc[:, ['category', 'tracer']]

    @property
    def size(self) -> int:
        return self.vectors.shape[0]
        
    @classmethod
    @debug.trace_args()
    def setup(cls, dconf: Dict | DictConfig, mapping: Mapping) -> "PriorConstraints":
        """
        Create a "PriorConstraints" object, which contains:
        - a dictionary with temporal correlation matrices
        - a dictionary with horizontal correlation matrices
        - a dictionary with standard deviations for each of the control variable
        - a DataFrame with the coordinates of the state variables (and any ancilliary data that may be needed).
        
        The information is taken from two sources:
        - the config dictionary (dconf), i.e. the "optimize" section of the main yaml file
        - the attribute of the "Category" objects, provided by the "Mapping" object
        - some functions provided by the "Mapping" object (i.e. "coarsen_cat")
        - the data in model space, passed via the "Mapping" object
        
        Dev note: initially, the idea was that the only argument would be the "Mapping" object, however, that proved too inflexible. Instead, it is better to pass as much as the configuration as possible via that "dconf" object (settings). The whole function may even be reworked to have three arguments: dconf, model_data (currently mapping.model_data) and the "coarsen_cat" function (currently mapping.coarsen_cat).
        """
        
        vectors = []
        sigmas, corr_t, corr_h = {}, {}, {}
        for cat in mapping.optimized_categories:
            # cat.horizontal_correlation_type = "e"
            # print( f'\ndconf.emissions = {dconf.emissions}' )
            # print( f'\ncat = {cat}' )
            catconf = dconf.emissions[cat.tracer]['categories'][cat.name]
            #match cat.error_structure.type:
            match cat.error_structure_type:
                case 'linear':
                    # Error, in the model space, is proportional to the absolute value of the flux
                    errmap = abs(mapping.model_data[cat.tracer][cat.name])
                    # Error, in the optim space, is obtained by aggregating model-space errors following the same approach as for aggregating fluxes themselves
                    errvec = mapping.coarsen_cat(cat, data=errmap.data, value_field='prior_uncertainty')
                case 'file':
                    ds = xr.open_dataset(catconf.error_structure.file)
                    errmap = (ds['dch4emis'] * ds.area * ds.timestep_length).values
                    errvec = mapping.coarsen_cat(cat, data=errmap**2, value_field='prior_uncertainty')
                    errvec.loc[:, 'prior_uncertainty'] = errvec.prior_uncertainty ** .5

            # Calculate (square root of the inverse of) the covariance matrices
            corr_t[cat] = calc_temporal_correlation(cat, errvec.loc[errvec.category == cat.name])
            corr_h[cat] = calc_horizontal_correlation(cat, errvec.loc[errvec.category == cat.name], cache_dir=dconf.get('cache_dir', None))
            
            # Calculate the total uncertainty, in any case
            if cat.get('annual_uncertainty') is not None :
                annual_budget = ureg(cat.annual_uncertainty)
                unit_budget = cat.total_uncertainty.units
            else :
                # Even if no key is provided, we calculate the total uncertainty (for information!). 
                # In this case, take the default unit from the "tracers" module.
                unit_budget = tracers.species[cat.tracer].unit_budget

            errtot = calc_total_uncertainty(errvec, corr_t[cat].B, corr_h[cat].B, cat.unit_optim, unit_budget)
            logger.info(f"Original uncertainty for category {cat.name}: {errtot:.3f} {unit_budget}")

            # Scale the sigmas to reach a target annual uncertainty (if it has been provided!):
            if cat.get('annual_uncertainty') is not None :
                # Deduce a scaling factor for the prior_uncertainty column:
                # Scale also by the simulation length, as uncertainty is provided in units/year
                nsec = errvec.loc[:, ['itime', 'dt']].drop_duplicates().dt.sum().total_seconds()
                scalef = (annual_budget.m / errtot) * (nsec / (365.25 * 86400))

                errvec.loc[:, 'prior_uncertainty'] *= scalef
                logger.info(f"Uncertainty for category {cat.name} set to {annual_budget.m} {annual_budget.u} (standard deviations scaled by {scalef = })")

            # If we optimize scaling factors, then the uncertainty needs to be divided by the prior
            if dconf.emissions.get('optim_scf') :
                # Set uncertainty to the ratio between uncertainty on the emissions and the emission themselves (in absolute values)
                # nan can occur when uncertainty on emissions is 0 ==> ensure it stays 0
                # inf can occur when emissions are 0 and uncertainty is not ==> set uncertainty to 0, the inversion would not be able to adjust it anyway
                prior_em = mapping.optim_data.loc[(mapping.optim_data.category == cat.name) & (mapping.optim_data.tracer == cat.tracer)].state_prior.values
                err_rel = (errvec.prior_uncertainty / prior_em).fillna(0).values
                err_rel[prior_em == 0] = 0
                errvec.loc[:, 'prior_uncertainty'] = abs(err_rel)
                
            # Store the results
            sigmas[cat] = errvec.prior_uncertainty.values
            vectors.append(errvec)

        vectors = concat(vectors)

        return cls(sigmas=sigmas, temporal_correlations=corr_t, horizontal_correlations=corr_h, vectors=vectors)

    @property
    def categories(self) -> List[Category]:
        return list(self.sigmas.keys())

    def save(self, dest: Path):
        for cat in self.categories:
            tbl = self.vectors.loc[(self.vectors.tracer == cat.tracer) & (self.vectors.category == cat.name)].to_xarray()
            tbl.attrs = cat.ncattrs
            tbl.to_netcdf(dest, group=f'{cat.tracer}.{cat.name}')
