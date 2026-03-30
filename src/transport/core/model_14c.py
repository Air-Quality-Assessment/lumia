#!/usr/bin/env python
from typing import List
from h5py import File
from numpy import zeros, nonzero, array, argsort
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass
from transport.emis import EmissionFields, Emissions
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List
import tempfile
import os
from lumia.utils import debug

from .model import Observations, shared_mem
from .model import Forward as BaseForward
from .model import Adjoint as BaseAdjoint
from .model import Model as BaseModel


@dataclass
class Forward(BaseForward):

    def run(self, emis: Emissions, obs: Observations) -> Observations :
        # Loop over the tracers:
        # The rational is that 2 tracers will likely have different set of footprints, while two categories for one tracer will share the same footprints

        fwd = self.run_tracer(emis, obs)

        # Combine :
        for col in [col for col in fwd.columns if col.startswith('mix')]:
            obs.loc[obs.index, col] = fwd.loc[:, col]

        return obs

    def run_tracer(self, emis: Emissions, obs: Observations) -> Observations:
        """
        Specific for CO2-14C dual tracer inversion.
        """

        # Retrieve the observations for that tracer, and their footprint file name:
        filenames = obs.footprint.dropna().drop_duplicates()

        # To optimize CPU usage in parallell simulations, process the largest files first
        nobs = array([obs.loc[obs.footprint == f].shape[0] for f in filenames])
        filenames = [filenames.values[i] for i in argsort(nobs)[::-1]]

        shared_mem.emis = emis
        shared_mem.obs = obs
        
        for obslist in self.run_files(filenames):
            for tr in obslist.tracer.dropna().unique() :
                for cat in emis[tr].categories : 
                    obs.loc[obslist.index, f'mix_{tr}_{cat}'] = obslist.loc[:, f'mix_{tr}_{cat}']#.astype(float)
            
                if tr == 'c14' :
                    for cat in emis['co2'].categories :
                        obs.loc[obslist.index, f'mix_{tr}_{cat}'] = obslist.loc[:, f'mix_{tr}_{cat}']

        shared_mem.clear('emis', 'obs')

        # Combine the flux components :
        try:
            obs.loc[:, 'mix'] = obs.mix_background.copy()
        except AttributeError:
            logger.warning(f'Missing background concentrations for tracer {emis.tracer}. Setting mix_background to 0')
            obs.loc[:, 'mix_background'] = 0
            obs.loc[:, 'mix'] = 0.

        for tr in emis.keys() :
            obs.loc[obs.tracer == tr, 'mix'] += obs.loc[obs.tracer == tr].filter(regex=f'mix_{tr}').sum(axis=1)

        return obs


    @staticmethod
    def run_file(filename: str, silent: bool = True) -> Observations: #dconf: Dict | DictConfig, 
        """
        Do a forward run on the selected footprint file. Set silent to False to enable progress bar. 
        Specific for CO2-14C dual tracer inversion.
        """
        obslist = shared_mem.obs

        obslist = obslist.loc[obslist.footprint == filename, ['obsid','tracer', 'mix_background_14c']]
        
        emis = shared_mem.emis

        with shared_mem.footprint_class(filename) as fpf :
            for iobs, obs in tqdm(obslist.iterrows(), desc=fpf.filename, total=obslist.shape[0], disable=silent):
                tr = obs.tracer
                # Align the coordinates
                fpf.align(emis[tr].grid, emis[tr].times.timestep, emis[tr].times.min)
                fp = fpf.get(obs.obsid)
                for cat in emis[tr].categories :
                    obslist.loc[iobs, f'mix_{tr}_{cat}'] = (emis[tr][cat].data[fp.itims, fp.ilats, fp.ilons] * fp.sensi).sum()
                
                if obs.tracer == 'c14':
                    for cat in emis['co2'].categories :
                        if 'is_natural' in emis['co2'][cat].attrs:
                            if emis['co2'][cat].is_natural:
                                obslist.loc[iobs, f'mix_co2_{cat}'] = (emis['co2'][cat].data[fp.itims, fp.ilats, fp.ilons] * fp.sensi).sum() # NOTE: used to convert to D14C permil
                                obslist.loc[iobs, f'mix_{tr}_{cat}'] = (emis['co2'][cat].data[fp.itims, fp.ilats, fp.ilons] * (obslist.loc[iobs, 'mix_background_14c']/1000) * fp.sensi).sum() 
                        if 'is_fossil' in emis['co2'][cat].attrs:
                            if emis['co2'][cat].is_fossil:
                                obslist.loc[iobs, f'mix_co2_{cat}'] = (emis['co2'][cat].data[fp.itims, fp.ilats, fp.ilons] * fp.sensi).sum() # NOTE: used to convert to D14C permil
                                obslist.loc[iobs, f'mix_{tr}_{cat}'] = (emis['co2'][cat].data[fp.itims, fp.ilats, fp.ilons] * -1 * fp.sensi).sum() 
        
        return obslist


class Adjoint(BaseAdjoint):

    @debug.timer
    def run(self, adj_emis: Emissions, obs: Observations) -> Emissions :
        return self.run_tracer(adj_emis, obs)

    def run_tracer(self, adjemis: EmissionFields, obs: Observations) -> EmissionFields :
        """
        Specific for CO2-14C dual tracer inversion.
        """

        # Retrieve the observations for that tracer, and their footprint file name:
        # Sort the files by the number of obs they contain (largest first)
        filenames = obs.footprint.dropna().drop_duplicates()
        nobs = array([obs.loc[obs.footprint == f].shape[0] for f in filenames])
        filenames = [filenames.values[i] for i in argsort(nobs)[::-1]]

        # Run the separate chunks
        shared_mem.obs = obs

        # Set the current data to 0:
        for adj in adjemis.tracers:
            adjemis[adj.tracer].setzero()

        # Get the shape of the adjoint field, store it in memory and create a new container for the data
        # For the CO2-14C case, we assume that all categories share the same grid and time axis, so we can just get it from the first category of the first tracer. This might need to be adapted for other cases.
        shared_mem.grid = adjemis[adj.tracer].grid
        shared_mem.time = adjemis[adj.tracer].times

        for adjfile in tqdm(self.run_files(filenames), desc='Concatenate adjoint files', leave=self.silent):
            
            with File(adjfile, 'r') as ds :
                components = {}
                for component in ['regular_co2', 'regular_c14', 'is_fossil', 'is_natural']:
                    if component in ds:
                        components[component] = {
                            'coords': ds[component]['coords'][:],
                            'values': ds[component]['values'][:]
                        }
                for adj in adjemis.tracers:
                    for cat in adj.categories:
                        arr = adjemis[adj.tracer][cat].data.reshape(-1)

                        # Regular adjoint contribution goes only to categories
                        # of the same tracer
                        regular_component = f"regular_{adj.tracer}"
                        if regular_component in components:
                            coords = components[regular_component]['coords']
                            values = components[regular_component]['values']
                            arr[coords] += values

                        # Special c14 -> co2 coupling terms:
                        # only CO2 categories flagged as fossil/natural receive them
                        if adj.tracer == 'co2':
                            if adj[cat].attrs.get('is_fossil', False):
                                coords = components['is_fossil']['coords']
                                values = components['is_fossil']['values']
                                arr[coords] += values

                            if adj[cat].attrs.get('is_natural', False):
                                coords = components['is_natural']['coords']
                                values = components['is_natural']['values']
                                arr[coords] += values
            os.remove(adjfile)

        # # Attempt of a new implementation using the multiprocessing.shared_memory module:
        # adjfield = adjemis[adjemis.categories[0]].data                                          # Just get the first category for reference (shape, size and dtype)
        # shm = shared_memory.SharedMemory(name='adjfield', create=True, size=adjfield.nbytes)    # Create the shared memory object
        # adjfield = ndarray(adjfield.shape, dtype=adjfield.dtype, buffer=shm.buf)                # Populate it with a numpy array
        # adjfield[:] = 0.                                                                        # Ensure the array is initialized with zeros
         
        # _ = tqdm(self.run_files(filenames), desc='Calculate adjoint chunks', leave=self.silent) # The subprocesses will then add data to it
        # for cat in adjemis.categories :
        #     #if adjemis[cat].optimized:
        #     adjemis[cat].data[:] = adjfield[:]
        #     #else :
        #     #    del adjemis[cat]
        # shm.close()
        shared_mem.clear('grid', 'time', 'obs')
        return adjemis

    @staticmethod
    def run_subset(filenames: List[str], silent: bool = True, tempdir: str = '/tmp') -> str :
        #observations = shared_memory.obs
        times = shared_mem.time
        grid = shared_mem.grid
        adj_emis = zeros((times.nt, grid.nlat, grid.nlon))
        adj_emis = {
            'regular_co2': adj_emis.copy(),
            'regular_c14': adj_emis.copy(),
            'is_fossil': adj_emis.copy(),
            'is_natural': adj_emis.copy()
        }

        for file in tqdm(filenames, disable=silent) :
            observations = shared_mem.obs.loc[shared_mem.obs.footprint == file]

            with shared_mem.footprint_class(file) as fpf :
                fpf.align(grid, times.timestep, times.min)

                for obs in tqdm(observations.itertuples(), desc=fpf.filename, total=observations.shape[0], disable=silent):
                    fp = fpf.get(obs.obsid) 
                    if obs.tracer == 'co2':
                        adj_emis['regular_co2'][fp.itims, fp.ilats, fp.ilons] += obs.dy * fp.sensi
                    elif obs.tracer == 'c14':
                        adj_emis['regular_c14'][fp.itims, fp.ilats, fp.ilons] += obs.dy * fp.sensi
                        adj_emis['is_natural'][fp.itims, fp.ilats, fp.ilons] += obs.dy * (obs.mix_background_14c/1000) * fp.sensi
                        adj_emis['is_fossil'][fp.itims, fp.ilats, fp.ilons] += obs.dy * -1 * fp.sensi

        with tempfile.NamedTemporaryFile(dir=tempdir, prefix='adjoint_', suffix='.h5') as fid :
            fname = fid.name
        with File(fname, 'w') as fid :
            sparse_adj_emis = {}
            for component, component_values in adj_emis.items():
                flat_values = component_values.reshape(-1)
                nz = nonzero(flat_values)[0]
                sparse_adj_emis[component] = {
                    'coords': nz,
                    'values': flat_values[nz]
                }

            for component, component_data in sparse_adj_emis.items():
                grp = fid.create_group(component)
                grp['coords'] = component_data['coords']
                grp['values'] = component_data['values']
        return fname

@dataclass
class Model(BaseModel):
    """Minimal adapter: reuse everything, just dispatch to 14C Forward/Adjoint."""

    def run_forward(self, obs: Observations, emis: Emissions) -> Observations :
        return Forward(self.footprint_class, self.parallel, self.ncpus, tempdir=self.tempdir).run(emis, obs)

    def run_adjoint(self, obs: Observations, adj_emis: Emissions) -> Emissions:
        return Adjoint(self.footprint_class, self.parallel, self.ncpus, tempdir=self.tempdir).run(adj_emis, obs)
