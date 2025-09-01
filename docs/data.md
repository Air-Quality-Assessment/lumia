# Emissions

The emissions to be transported are defined in the `emissions` section of the [configuration file](settings.md). The section is structured as follows:
```yaml
emissions :
    tracer : [tracer_i, tracer_j, ...] # List of tracers 
    tracer_i:
        region :        # Grid definition for tracer "tracer_i" 
        interval :      # transport model temporal resolution
        categories :
            cat1 :      # "origin" of the emission files for cat "cat1"
            cat2 :
                origin : # "origin" of the emission files for cat "cat2"
                ...    : # other settings, speficic to cat2
            ...
        metacategories :
            metacat1 : 
            metacat2 :
        ...            :  
    tracer_j:
        ...
```

This section of the configuration file is essentially used to construct a single LUMIA emission file, containing the emissions for the simulation (i.e. emissions of all categories and all tracers, covering the entire length of the simulation), based on category and tracer-specific pre-processed emission files (see File formats section below).

The yaml file keys determine the following settings:

- Mandatory keys:
    - list of emission tracers: **emissions.tracers** key (should be a list!)
    - path of the pre-processed files to be read: **path**, **prefix** and **origin** keys. 
    - grid definition of the emission files : **emissions.{tracer}.region** key
    - temporal resolution of the emissions: **emissions.{tracer}.interval** key
- Optional keys:
    - name of the netCDF variable to be read: **field** key
    - whether the pre-processed emission files should be resampled from a different temporal resolution: **resample_from** key.
    - path to a rclone-mounted emission archive: **archive** key
    - definition of metacategories: **emissions.{tracer}.metacategories** key(s)

The **path**, **prefix**, **field**, **resample_from** and **archive** keys can be provided at the tracer level (e.g. **emissions.{tracer}.path**) or at the category level (**emissions.{tracer}.categories.{catname}.path**). The **origin** key should be provided at the category level (**emissions.{tracer}.categories.{catname}.origin**), but if all the other keys are provided at the tracer-level, it is possible to simply use **emissions.{tracer}.categories.{catname}** as an "origin" key.

## Pre-processed emission files

The assembling of pre-processed, category-specific emission files into a single, simulation-specific emission file is performed by the [`lumia.Data.from_dconf` method](https://github.com/lumia-dev/lumia/blob/master/src/lumia/data/xr.py).

The files should contain three coordinate variables (*time*, *lat* and *lon*), one (or more) emission fields defined on the same coordinates. It is also recommended to include a (*lat*, *lon*) area field for convenience, but this is not read or required by LUMIA. The *lat* and *lon* variables should refer to the center of the grid cells, while the *time* coordinate points to the start of each time period. The *time* coordinate should contain integer, and have a *units* and a *calendar* attribute, allowing conversion to a `numpy.datetime64` type. The recommended method for creating these files is to use the [xarray](https://xarray.dev/) library.

The path of the pre-processed files is determined by the **path**, **prefix** and **origin** keys: The file names follows the pattern **{path}/{tres}/{prefix}{origin}.*.nc**. Here **prefix** is meant as a tracer-spefic prefix (e.g. "co2_flux") and **origin** refers to the origin of the data in the file (e.g. "LPJ"). The **{tres}** key refers to the temporal resolution of the pre-processed emission files. By default it is identical as **emissions.{tracer}.interval**, but can be set at a lower temporal resolution using the **resample_from** key (in which case the emissions will simply be rebinned by LUMIA). 

For instance, if a **biosphere** category should read emissions from */data/LUMIA/M/co2_emis.LPJGUESS-v20.2018.nc*, then:

- **emissions.co2.path** (or **emissions.co2.categories.biosphere.path**) should be set to */data/LUMIA*
- **emissions.co2.prefix** (or **emissions.co2.categories.biosphere.prefix**) should be set to *co2_emis.*
- **emissions.co2.categories.biosphere.origin** (or **emissions.co2.categories.biosphere**) should be set to *LPJGUESS-v20*
- if **emissions.co2.interval** is not set to *M*, then a **emissions.co2.resample_from** (or **emissions.co2.categories.biosphere.resample_from**) key should be defined and set to *M*.
- if the file contains any other variable than the coordinates (*lat*, *lon*, *time*) and the variable containing the emission themselves, then a **emissions.co2.categories.biosphere.field** variable should be set.

Note that at no point we specify the time component of the filename: LUMIA will load all the files matching the pattern **{path}/{prefix}{origin}.*.nc** as a [multi-file netCDF Dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html).

## Archive

The emission files need to be on a local file system for LUMIA to read them, however that can be just a temporary folder, with the files being stored on a remote permanent storage, accessed via [rclone](https://rclone.org/). The path to the archive should be provided by the `emissions.{tracer}.archive` key, with the pattern `rclone:{rclone_remote}:path/to/remote/dir`, where `{rclone_remote}` is a rclone remote path defined in your `rclone.conf` file (see the relevant [rclone documentation](https://rclone.org/docs/#syntax-of-remote-paths)).

## Meta-categories

It is possible to define *meta-categories*, as a linear combination of other categories. For instance in the (fictional) example below, we defined one "NEE" meta-category, as the sum of the GPP and respiration categories. We defined a "fossil" category by subtraction of "agri_fires" from the "anthropogenic" category, and we created a "fires" category combining the "natural_fires" one and the "agri_fires" one, but the latter with a 1.2 scaling factor.

By default, the meta-categories are not transported (i.e. the transport model is unaware of them), but their impact on the concentrations is calculated. On the other hand, the categories used to build the meta-categories are transported, but their impact on concentrations is ignored.

```yaml
emissions :
    co2 :
        categories:
            GPP : LPJ_GPP
            respiration : LPJ_resp
            anthropogenic : TNO
            natural_fires : GFAS
            agri_fires : 
                origin : EDGAR
                field : agriwasteburning
        meta-categories :
            NEE : GPP + respiration
            fossil : anthropogenic - agri_fires
            fires : natural_fires + 1.2 * agri_fires
```

## LUMIA emission file

```python
raise SectionYetToBeWrittenError ;-)
```