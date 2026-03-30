#!/usr/bin/env python
import shutil
from dataclasses import dataclass, field
from pandas import DataFrame, Timestamp, read_csv, read_hdf
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List
import tarfile
from loguru import logger
import tempfile
import os
from numpy.typing import NDArray
from datetime import datetime
from lumia.utils import debug

from .observations import Settings as BaseSettings
from .observations import Observations as BaseObservations

@dataclass
class Settings(BaseSettings):
    err_min : float = 0
    err_freq : str = '7D'
    err_fac : float = 1
    chi2v_target: float = 6
    field_err_obs : str = 'err_obs'
    field_bg_14c : str = 'background_14c'

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, v)


@dataclass
class Observations14C(BaseObservations):

    @classmethod
    def from_dconf(cls, dconf: Dict | DictConfig) -> "Observations":
        """
        Generate an `Observations` object based on a dictionary of keys.
        The corresponding yaml section should have the following structure:
        
        {
            file : 
            start :
            end :
            rename :
                field1 : field1_newname
                field2 : field2_newname
            uncertainties :
                err_min :
                err_freq :
                err_fac :
                field_err_obs :
        }
        
        Apart from "file", the other sections/keys are optional:
        - start and end can be used to restrict the time-span of the obs database
        - the rename key is used to specify a list of columns to be renamed
        - the uncertainties section is used to pass settings for computing the obs uncertainties (see Observations.calc_uncertainties method).
        """
        obs = cls.from_tar(dconf['file'])
        
        # Restrict the list of observations to a certain time range:
        obs.select_times(dconf.get('start'), dconf.get('end'))

        # Rename columns (if needed!)
        obs.observations.rename(columns=dconf.get('rename', {}), inplace=True)
        
        # Set uncertainty settings
        obs.settings.update(**dconf.get('uncertainties', {}))
        
        return obs

        
    @debug.trace_args()
    def calc_uncertainties(self, step: str = None):
        
        # Ensure that all observations have measurement error:
        sel = self.observations.loc[:, self.settings.field_err_obs] <= 0
        self.observations.loc[sel, self.settings.field_err_obs] = self.settings.err_min

        tracers = self.observations['tracer'].unique()

        for tr in tracers:
            types = self.observations.loc[self.observations.tracer == tr, "type"].unique()
            for tp in types:
                for code in self.observations.code.drop_duplicates():

                    mask = (self.observations.tracer == tr) & (self.observations.type == tp) & (self.observations.code == code)
                    
                    # 1) Select the data:
                    mix = self.observations.loc[mask].loc[:, ['time', 'obs', f'mix_{step}', self.settings.field_err_obs]].set_index('time').sort_index()
                    
                    # 2) Calculate weekly moving average and residuals from it
                    #trend = mix.rolling(freq).mean()
                    #resid = mix - trend
                    
                    # Use a weighted rolling average, to avoid giving too much weight to the uncertain obs:
                    weights = 1. / mix.loc[:, self.settings.field_err_obs] ** 2
                    total_weight = weights.rolling(self.settings.err_freq).sum()   # sum of weights in a week (for normalization)
                    obs_weighted = mix.obs * weights
                    mod_weighted = mix.loc[:, f'mix_{step}'] * weights
                    obs_averaged = obs_weighted.rolling(self.settings.err_freq).sum() / total_weight
                    mod_averaged = mod_weighted.rolling(self.settings.err_freq).sum() / total_weight
                    resid_obs = mix.obs - obs_averaged
                    resid_mod = mix.loc[:, f'mix_{step}'] - mod_averaged
                    
                    # 3) Calculate the standard deviation of the residuals model-data mismatches. Store it in sites dataframe for info.
                    sigma = (resid_obs - resid_mod).dropna().values.std()
                    self.sites.loc[self.sites.code == code, 'err'] = sigma
                    logger.info(f'Model uncertainty for site {code} set to {sigma:.2f}')
                    
                    # 4) Get the measurement uncertainties and calculate the error inflation
        #            s_obs = self.observations.loc[:, self.settings.field_err_obs].values
        #            nobs = len(s_obs)
        #            s_mod = sqrt((nobs * sigma**2 - (s_obs**2).sum()) / nobs)

                    # 5) Store the inflated errors:
                    self.observations.loc[mask, 'err'] = (
                        self.observations.loc[mask, self.settings.field_err_obs] ** 2 + sigma ** 2).values ** .5
                    self.observations.loc[mask, 'resid_obs'] = resid_obs.values
                    self.observations.loc[mask, 'resid_mod'] = resid_mod.values
                    self.observations.loc[mask, 'obs_detrended'] = obs_averaged.values
                    self.observations.loc[mask, 'mod_detrended'] = mod_averaged.values
        
                    # Apply global scaling factor if needed:
                    chi2v = self.reduced_chi2(mask, step)

                    if abs(chi2v - self.settings.chi2v_target) >= 0.5:
                        self.settings.err_fac = float((chi2v / self.settings.chi2v_target)**0.5)

                    self.observations.loc[mask, 'err'] *= self.settings.err_fac

                    chi2v = self.reduced_chi2(mask, step)

                    logger.info(f'Type: {tp} Tracer: {tr} Reduced Chi-squared ({step}): {chi2v}, with a scaling factor of {self.settings.err_fac}.')


    def reduced_chi2(self, mask, step):

        # Calculate chi squared
        obs = self.observations.loc[mask, 'obs'].values

        mod = self.observations.loc[mask, f'mix_{step}'].values

        sigma = self.observations.loc[mask, 'err'].values

        chi2 = sum(((obs - mod) / sigma) ** 2)

        return chi2 / (len(obs) - 1)

                