#!/usr/bin/env python

from argparse import ArgumentParser
import sys
from pathlib import Path
from pandas import Timestamp
import lumia


p = ArgumentParser()
p.add_argument('--machine', '-m', default='donkey')
p.add_argument('config', type=Path)
p.add_argument('--verbosity', '-v', default=lumia.setup_logging('DEBUG'), type=lumia.setup_logging)
p.add_argument('--start')
p.add_argument('--end')
p.add_argument('--spinup', default=None, type=str)
p.add_argument('--spindown', default=None, type=str)


args = p.parse_args(sys.argv[1:])


# Load the config file:
dconf = lumia.read_config(
    args.config, 
    machine=args.machine, 
    run={'start': args.start, 'end':args.end, 'spinup': args.spinup, 'spindown': args.spindown},
)


# Append the start and end times to the tag:
#dconf.run.tag = f'{dconf.run.tag}/{Timestamp(dconf.run.start):%Y%m%d}-{Timestamp(dconf.run.end):%Y%m%d}'


# Save the settings
lumia.settings.write(dconf, Path(dconf.run.paths.output) / 'config.yaml')


# Setup the observations
obs = lumia.Observations.from_dconf(dconf.observations)
obs.observations.assim = False
obs.observations.assim = False

if dconf.observations.filter_by_hour:
    for site in obs.sites.itertuples():
        if site.assim_start > site.assim_end:
            # Night time data
            obs.observations.loc[(obs.observations.site == site.Index) & ((obs.observations.time.dt.hour >= site.assim_start) | (obs.observations.time.dt.hour <= site.assim_end)), 'assim'] = True
        else :
            # Day time data
            obs.observations.loc[(obs.observations.site == site.Index) & (site.assim_start <= obs.observations.time.dt.hour) & (site.assim_end >= obs.observations.time.dt.hour), 'assim'] = True
    obs.observations = obs.observations[obs.observations.assim]


# Setup the emissions:
emis = lumia.Data.from_dconf(dconf, dconf.run.start, dconf.run.end)


# # Setup the transport model
transport = lumia.Transport(**dconf.model)
# transport.setup_observations(obs)
# transport.calc_departures(emis, step='apri')


# Setup the prior/mapping
mapping = lumia.Mapping.init(dconf, emis)
prior = lumia.PriorConstraints.setup(dconf.optimize, mapping)


opt = lumia.Optimizer(#optimizer.OptimLanczos(
    prior = prior,
    model = transport,
    mapping = mapping,
    observations = obs,
    settings = dconf.run.optimizer
)
x_opt = opt.solve()
opt.vectors.loc[:, 'state_preco_apos'] = x_opt
opt.vectors.loc[:, 'state_apos'] = opt.xc_to_x(x_opt)
apos = mapping.vec_to_struct(opt.vectors.state_apos.values)
# opt.vectors.to_xarray().to_netcdf(Path(dconf.run.paths.output) / 'states.nc')
apos.convert('nmol / m**2 / s')
apos.to_netcdf(Path(dconf.run.paths.output) / 'emissions.apos.nc', zlib=True)
obs.save_tar(Path(dconf.run.paths.output) / 'observations.apos.tar.gz')
