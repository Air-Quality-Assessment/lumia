"""
save_and_plot_clusters.py
-------------------------
Two self-contained utilities:

    save_clusters(clusters_out, path, grid, sensi_map=None)
        Save cluster specs (and optionally the sensitivity map) to a single
        NetCDF file using xarray.

    load_and_plot_clusters(nc_path, out_path=None, n_label=None)
        Load the NetCDF and produce the two-panel sensitivity / cluster figure.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_clusters(
    clusters_out : List[Dict],
    path         : str | Path,
    grid,                           # lumia Grid object — provides nlat, nlon, latc, lonc
    sensi_map    : np.ndarray = None,
    tracer       : str = "co2",
    category     : str = "biosphere",
) -> Path:
    """
    Save cluster specs to a NetCDF file.

    The ragged per-cluster membership (ilats, ilons) is stored in a flat
    layout with an offset index — no padding, no wasted space.

    Variables written
    -----------------
    flat_ilats / flat_ilons  : all grid-point indices concatenated
    offsets                  : start position of each cluster in the flat arrays
    sizes                    : number of grid points in each cluster
    lat_centre / lon_centre  : area-weighted centre of each cluster
    sensi                    : (optional) sensitivity map on the original grid

    Parameters
    ----------
    clusters_out : list of dicts with keys ilats, ilons, lats, lons
    path         : output NetCDF path
    grid         : lumia Grid (or any object with nlat, nlon, latc, lonc)
    sensi_map    : 2-D ndarray (nlat, nlon), optional
    """
    path = Path(path)
    n_clusters = len(clusters_out)

    # --- flat layout ---
    all_ilats = np.concatenate([c["ilats"] for c in clusters_out]).astype(np.int32)
    all_ilons = np.concatenate([c["ilons"] for c in clusters_out]).astype(np.int32)
    sizes     = np.array([len(c["ilats"]) for c in clusters_out], dtype=np.int32)
    offsets   = np.concatenate([[0], sizes.cumsum()[:-1]]).astype(np.int32)

    lat_centre = np.array([c["lats"].mean() for c in clusters_out], dtype=np.float32)
    lon_centre = np.array([c["lons"].mean() for c in clusters_out], dtype=np.float32)

    ds = xr.Dataset(
        {
            "flat_ilats":  (["flat_point"],  all_ilats),
            "flat_ilons":  (["flat_point"],  all_ilons),
            "offsets":     (["cluster"],     offsets),
            "sizes":       (["cluster"],     sizes),
            "lat_centre":  (["cluster"],     lat_centre),
            "lon_centre":  (["cluster"],     lon_centre),
        },
        attrs={
            "description": f"Spatial cluster specs — {tracer}/{category}",
            "tracer":       tracer,
            "category":     category,
            "n_clusters":   n_clusters,
            "nlat":         grid.nlat,
            "nlon":         grid.nlon,
        },
    )

    # Store the grid coordinates so the file is self-contained
    ds["lat"] = xr.DataArray(grid.latc.astype(np.float32), dims=["lat_grid"],
                             attrs={"units": "degrees_north"})
    ds["lon"] = xr.DataArray(grid.lonc.astype(np.float32), dims=["lon_grid"],
                             attrs={"units": "degrees_east"})

    # Attribute metadata
    ds["flat_ilats"].attrs = {"long_name": "Latitude index in the model grid"}
    ds["flat_ilons"].attrs = {"long_name": "Longitude index in the model grid"}
    ds["offsets"].attrs    = {"long_name": "Start index of each cluster in the flat arrays"}
    ds["sizes"].attrs      = {"long_name": "Number of grid points in each cluster"}
    ds["lat_centre"].attrs = {"units": "degrees_north", "long_name": "Cluster centre latitude"}
    ds["lon_centre"].attrs = {"units": "degrees_east",  "long_name": "Cluster centre longitude"}

    if sensi_map is not None:
        ds["sensi"] = xr.DataArray(
            sensi_map.astype(np.float32),
            dims=["lat_grid", "lon_grid"],
            attrs={"long_name": "Network sensitivity", "units": "s m2 mol-1"},
        )

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    print(f"Saved {n_clusters} clusters → {path}")
    return path


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------

def load_clusters(path: str | Path) -> tuple[xr.Dataset, List[Dict]]:
    """
    Load a cluster NetCDF and return the Dataset plus a list-of-dicts
    (same format as clusters_out) for convenient iteration.
    """
    ds = xr.open_dataset(path)

    flat_ilats = ds["flat_ilats"].values
    flat_ilons = ds["flat_ilons"].values
    offsets    = ds["offsets"].values
    sizes      = ds["sizes"].values
    lats       = ds["lat"].values
    lons       = ds["lon"].values

    clusters = []
    for i in range(ds.attrs["n_clusters"]):
        s = offsets[i]
        n = sizes[i]
        il = flat_ilats[s : s + n]
        io = flat_ilons[s : s + n]
        clusters.append({
            "ilats": il,
            "ilons": io,
            "lats":  lats[il],
            "lons":  lons[io],
        })

    return ds, clusters


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clusters(
    ds         : xr.Dataset,
    clusters   : List[Dict],
    out_path   : str | Path | None = None,
    n_label    : str | None = None,
    dpi        : int = 300,
    res        : float = 0.1,          # grid resolution in degrees (for Rectangle width)
) -> plt.Figure:
    """
    Reproduce the two-panel sensitivity / cluster figure.

    Panel a — sensitivity map (log scale, viridis)
    Panel b — cluster bounding boxes coloured randomly

    Parameters
    ----------
    ds        : Dataset returned by load_clusters (must contain 'sensi')
    clusters  : list of dicts from load_clusters
    out_path  : save path (PNG/PDF). If None, the figure is shown interactively.
    n_label   : label for the cluster count in the title, e.g. "8 000"
    res       : native grid resolution in degrees (used for the Rectangle nudge)
    """
    extent   = [float(ds.attrs.get("lon0", ds["lon"].values.min() - res/2)),
                float(ds.attrs.get("lon1", ds["lon"].values.max() + res/2)),
                float(ds.attrs.get("lat0", ds["lat"].values.min() - res/2)),
                float(ds.attrs.get("lat1", ds["lat"].values.max() + res/2))]

    has_sensi = "sensi" in ds

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax0, ax1 = axes

    # ------------------------------------------------------------------
    # Panel a — sensitivity map
    # ------------------------------------------------------------------
    if has_sensi:
        sensi_val = ds["sensi"].values
        sensi_pos = np.where(sensi_val > 0, sensi_val, np.nan)

        vmin = np.nanmin(sensi_pos)
        vmax = np.nanmax(sensi_pos)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        im = ax0.imshow(
            np.log(np.where(sensi_val > 0, sensi_val, np.nan)),
            extent=extent,
            origin="lower",
            cmap="viridis",
            transform=ccrs.PlateCarree(),
        )
        ax0.coastlines(linewidth=0.8)
        ax0.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax0.set_title("Network sensitivity to fluxes (log scale)", fontsize=11)

        cb = fig.colorbar(
            mcm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=ax0, orientation="horizontal", pad=0.05, fraction=0.046,
        )
        cb.set_label(r"Sensitivity,  s m$^2$ mol$^{-1}$", fontsize=10)
    else:
        ax0.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                 transform=ax0.transAxes, fontsize=12)
        ax0.set_title("Sensitivity map (not available)", fontsize=11)

    # ------------------------------------------------------------------
    # Panel b — cluster bounding boxes
    # ------------------------------------------------------------------
    # rng   = np.random.default_rng(seed=42)   # reproducible colours
    # cmap  = plt.get_cmap("tab20")

    for cl in clusters:
        # color = cmap(rng.integers(0, 20))
        lon_min, lon_max = cl["lons"].min(), cl["lons"].max()
        lat_min, lat_max = cl["lats"].min(), cl["lats"].max()

        ax1.add_patch(
            mpatches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min + res,
                lat_max - lat_min + res,
                fill        = False,
                # facecolor   = color,
                edgecolor   = "k",
                linewidth   = 0.3,
                # alpha       = 0.6,
                transform   = ccrs.PlateCarree(),
            )
        )

    ax1.add_feature(cfeature.OCEAN,  zorder=2, facecolor="lightgrey", edgecolor="k")
    ax1.add_feature(cfeature.BORDERS, zorder=3, linewidth=0.4, linestyle=":")
    ax1.coastlines(linewidth=0.8, zorder=3)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())

    n_str  = n_label or f"{len(clusters):,}"
    tracer = ds.attrs.get("tracer", "")
    cat    = ds.attrs.get("category", "")
    ax1.set_title(f"Optimized clusters  n = {n_str}  ({tracer}/{cat})", fontsize=11)

    # Panel labels
    for ax, label in zip(axes, ("a)", "b)")):
        ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved → {out_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def load_and_plot_clusters(
    nc_path  : str | Path,
    out_path : str | Path | None = None,
    n_label  : str | None = None,
    res      : float = 0.1,
) -> plt.Figure:
    """Load a cluster NetCDF and produce the two-panel figure in one call."""
    ds, clusters = load_clusters(nc_path)
    return plot_clusters(ds, clusters, out_path=out_path, n_label=n_label, res=res)


# ---------------------------------------------------------------------------
# Example usage (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python save_and_plot_clusters.py <clusters.nc> [output.png]")
        sys.exit(0)

    nc   = sys.argv[1]
    png  = sys.argv[2] if len(sys.argv) > 2 else None
    load_and_plot_clusters(nc, out_path=png)