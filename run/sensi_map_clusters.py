"""
sensi_map_clusters.py
---------------------
Computes (or reloads) the sensitivity map, derives spatial clusters via the lumia Mapping class, 
saves them to a self-contained NetCDF, and produces a
two-panel sensitivity / cluster figure.

Set REUSE_SENSI = True to skip the transport run when the sensitivity map
has already been saved from a previous run.

Usage
-----
    python sensi_map_clusters.py
"""

import os
import time
import numpy as np
import xarray as xr
from pathlib import Path
from pandas import Timestamp
from pandas.tseries.frequencies import to_offset
from loguru import logger
from tqdm import tqdm

os.environ["NUMEXPR_MAX_THREADS"] = "24"

import lumia
from save_and_plot_clusters import save_clusters, load_and_plot_clusters

start_time = time.time()

# ===========================================================================
# CONFIG
# ===========================================================================

REUSE_SENSI = True      # ← set False to recompute the sensitivity map

RC_PATH  = '/lunarc/nobackup/projects/ghg_inv/carlos/scripts/scripts_lumia_new/yaml'
MACHINE  = 'cosmos_i'
TRACER   = 'co2'
CATEGORY = 'anthro'
GRID_RES = 0.25          # degrees — used for the cluster bounding-box nudge in the figure

# ===========================================================================
# END CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Read configuration
# ---------------------------------------------------------------------------
conf = lumia.read_config(os.path.join(RC_PATH, 'Test.yaml'), machine=MACHINE)
lumia.settings.write(conf, Path(conf.run.paths.output) / 'config.yaml')

# spinup   = to_offset(conf.run.get('spinup',   '0h'))
# spindown = to_offset(conf.run.get('spindown', '0h'))
# conf.run.start = str(Timestamp(conf.run.start) - spinup)
# conf.run.end   = str(Timestamp(conf.run.end)   + spindown)

out_dir    = Path(conf.run.paths.output)
n_clusters = str(conf.optimize.emissions[TRACER][CATEGORY].npoints)

SENSI_PATH   = out_dir / 'sensi_map.nc'
CLUSTER_PATH = out_dir / f'cluster_specs_{n_clusters}.nc'
FIGURE_PATH  = out_dir / f'Clusters_{n_clusters}.png'

# ---------------------------------------------------------------------------
# Emissions  (always needed for grid / temporal mapping)
# ---------------------------------------------------------------------------
logger.info("Loading emissions...")
emis = lumia.Data.from_dconf(conf, conf.run.start, conf.run.end)

# ---------------------------------------------------------------------------
# Sensitivity map
# ---------------------------------------------------------------------------
if REUSE_SENSI and SENSI_PATH.exists():
    logger.info("Loading existing sensitivity map: %s", SENSI_PATH)
    sensi_ds  = xr.open_dataset(SENSI_PATH)
    sensi_map = {tr: sensi_ds[tr].values for tr in sensi_ds.data_vars}

else:
    logger.info("Running transport model to compute sensitivity map...")

    obs = lumia.Observations14C.from_tar(conf.observations.file.path)
    obs.select_times(tmin=conf.run.start, tmax=conf.run.end, inplace=True)
    obs.observations.rename(columns={conf.observations.field_bg_14c: 'mix_background_14c'}, inplace=True)

    transport = lumia.Transport14C(**conf.model)

    # brings model to vector 
    mapping = lumia.Mapping.init(conf, emis)

    prior = lumia.PriorConstraints.setup(conf, mapping)

    #Setup the optimizer
    opt = lumia.optimizer.OptimLanczos14C(
        prior=prior, 
        model=transport, 
        mapping=mapping, 
        observations=obs, 
        settings=conf.run.optimizer
        )

    logger.info("Calculating sensitivity map...")
    sensi_map = transport.calc_sensi_map(emis)

    # Save for reuse
    ds_sensi = xr.Dataset({tr: xr.DataArray(v, attrs={'units': 's m2 mol-1'})
                           for tr, v in sensi_map.items()})
    ds_sensi.to_netcdf(SENSI_PATH)
    logger.info("Sensitivity map saved: %s", SENSI_PATH)

    # Clean up transport scratch files
    for fname in ['adjoint.nc', 'adjoint.rc', 'departures.tar.gz']:
        fpath = out_dir / fname
        if fpath.exists():
            fpath.unlink()

# ---------------------------------------------------------------------------
# Spatial clusters via Mapping.init
# (skips SetupPrior / SetupUncertainties — clusters only)
# ---------------------------------------------------------------------------
logger.info("Computing spatial clusters via Mapping.init...")
t0 = time.perf_counter()

mapping = lumia.Mapping.init(conf, emis, sensi_map)

logger.info("Mapping initialised in %.1f s", time.perf_counter() - t0)

# ---------------------------------------------------------------------------
# Extract cluster specs from spatial_mapping
#
# spatial_mapping[cat].overlap_fraction has shape (n_model_points, n_optim_points).
# Non-zero entries in column icl identify the model grid points of cluster icl.
# Flat indices → (ilat, ilon) via:  ilat = flat // nlon,  ilon = flat % nlon
# ---------------------------------------------------------------------------
clusters_out = []
 
for cat in mapping.optimized_categories:
    grid      = emis[cat.tracer].grid
    nlat, nlon = grid.nlat, grid.nlon
 
    sm = mapping.spatial_mapping[cat]
    of = sm['overlap_fraction'].values        # (n_model_points, n_optim_points)
 
    logger.info(
        "Extracting cluster specs for %s / %s  (%d clusters)",
        cat.tracer, cat.name, of.shape[1],
    )
 
    for icl in tqdm(range(of.shape[1]), desc=f"Extracting {cat.name}"):
        flat_idx = np.where(of[:, icl] != 0)[0]
        ilats    = (flat_idx // nlon).astype(np.int32)
        ilons    = (flat_idx %  nlon).astype(np.int32)
        clusters_out.append({
            "tracer":   cat.tracer,
            "category": cat.name,
            "ilats":    ilats,
            "ilons":    ilons,
            "lats":     grid.latc[ilats].astype(np.float32),
            "lons":     grid.lonc[ilons].astype(np.float32),
        })
 
logger.info("Extracted %d clusters total", len(clusters_out))

# ---------------------------------------------------------------------------
# Save clusters to NetCDF  (one file per tracer/category)
# ---------------------------------------------------------------------------
for tracer, category in set((c["tracer"], c["category"]) for c in clusters_out):
    cat_clusters = [c for c in clusters_out
                    if c["tracer"] == tracer and c["category"] == category]
    n = len(cat_clusters)
 
    nc_path  = out_dir / f"cluster_specs_{tracer}_{category}_{n}.nc"
    fig_path = out_dir / f"Clusters_{tracer}_{category}_{n}.png"
 
    logger.info("Saving %d clusters for %s/%s → %s", n, tracer, category, nc_path)
    save_clusters(
        cat_clusters,
        path      = nc_path,
        grid      = emis[tracer].grid,
        sensi_map = sensi_map.get(tracer),
        tracer    = tracer,
        category  = category,
    )
 
    logger.info("Generating figure for %s/%s...", tracer, category)
    load_and_plot_clusters(
        nc_path  = nc_path,
        out_path = fig_path,
        n_label  = f"{n:,}",
        res      = GRID_RES,
    )

logger.info("=== Done in {:.2f} minutes ===", (time.time() - start_time) / 60)