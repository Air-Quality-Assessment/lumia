#!/usr/bin/env python

import os
import shutil
from pandas import DataFrame, read_hdf, isnull
from numpy import array, nan
from tqdm import tqdm
from loguru import logger
from transport.core.model import FootprintFile
from typing import Type
import xarray as xr
from transport.concentrations import interp_file
from pathlib import Path

from .observations import Observations as BaseObservations


common = {}


def check_migrate(source, dest):
    if os.path.exists(dest):
        return True
    elif os.path.exists(source):
        shutil.copy(source, dest)
        return True
    return False


class Observations(BaseObservations):

    def find_footprint_files(self, archive: str, local: str=None) -> None:
        # 2) Append file names, both for local and archive
        if local is None:
            local = archive
        
        fnames_archive = self.footprint.copy()
        fnames_local = self.footprint.copy()

        for tp in self.type.unique():

            logger.info(f'Searching footprints in {archive[tp]} for type {tp}')

            mask = self.type == tp
            fnames_archive[mask] = archive[tp] + '/' + self.footprint[mask]
            fnames_local[mask] = local[tp] + '/' + self.footprint[mask]

            # 3) retrieve the files from archive if needed:
            if local is not None :
                Path(local[tp]).mkdir(exist_ok=True, parents=True)
        
        exists = array([check_migrate(arc, loc) for (arc, loc) in tqdm(zip(fnames_archive, fnames_local), desc='Migrate footprint files', total=len(fnames_local), leave=False)])
        self.loc[:, 'footprint'] = fnames_local

        for tp in self.type.unique():
            mask = self.type == tp
            if not exists[mask].any():
                logger.warning(f"No valid footprints found for type {tp}.")
                raise RuntimeError("No valid footprints found")
            else:
                logger.info(f"Found {exists[mask].sum()} valid footprints for type {tp} out of {len(exists[mask])} observations.")

        # if not exists.any():
        #     logger.error("No valid footprints found. Exiting ...")
        #     raise RuntimeError("No valid footprints found")

        # Create the file names:
        #fnames = path + '/' + self.code.str.lower() + self.height.map('.{:.0f}m.'.format) + self.time.dt.strftime('%Y-%m.hdf')
        #self.loc[:, 'footprint'] = fnames
        #exists = array([os.path.exists(f) for f in fnames])
        self.loc[~exists, 'footprint'] = nan


    def interp_background(self, conc_field : xr.Dataset, footprint_class: Type[FootprintFile]): 
        # TODO: implement a parallel version of this function and a version for 14C-specific footprints.
        # single process implementation: 
        files = self.loc[:, ['footprint', 'tracer']].drop_duplicates().dropna()
        self.loc[:, 'mix_background'] = nan
        pbar = tqdm(files.itertuples(), total=len(files))
        for fpfile in pbar:
            obs = self.loc[(self.footprint == fpfile.footprint) & (self.tracer == fpfile.tracer), ['footprint', 'obsid', 'tracer']]
            with footprint_class(fpfile.footprint) as fpf:
                endpoints = fpf.get_endpoints(obs.obsid)
            bg = interp_file(conc_field, endpoints, fpfile.tracer.upper(), field='mix_interpolated')
            tqdm.write(f'Mean {fpfile.tracer} background for file {fpfile.footprint}: {bg.mix_interpolated.mean()}')
            bg = self.merge(bg, on='obsid', how='left').set_index(self.index).mix_interpolated.dropna()
            self.loc[bg.index, 'mix_background'] = bg