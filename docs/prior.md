# Prior uncertainties 

The uncertainty settings should be grouped in a section of the configuration file, with the following structure :

```yaml
optimize :
    emissions :
        tracer1 :
            category1 :
                annual_uncertainty :
                spatial_correlation :
                    cortype :
                temporal_correlation :
                optimization_interval :
            category2 : 
                ...
        tracer2 :
            ...
```

The **annual_uncertainty** settings should be in the form of a value and unit (e.g. "1.2 TgCH4", "0.5 PgCO2", ...). The **optimization_interval** settings should be a string understandable by the [`pandas.tseries.frequencies.to_offset`](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html) function. The **spatial_correlation** and **temporal_correlation** should contain a `cortype` sub-key, which determines which correlation function should be used. Depending on this, they may require additional subkeys (e.g. `corlen`, `stretch_ratio`, etc.).

## Error-covariance matrix

The error covariance matrix is calculated in three steps:

1. The standard are estimated, for each state vector component
2. The correlations are determned
3. The resulting total uncertainty is computed (by combining the standard deviations and correlations), and the standard deviations are scaled uniformly, to match a user-specified annual uncertainty target.

???+ note "where is it in the code?"
    The uncertainty matrix is calculated in the [`lumia.PriorConstraints.setup`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/prior/prior.py) method, which should be called by the main script (see, e.g. [run/co2_inversion.py](https://github.com/lumia-dev/lumia/blob/master/run/co2_inversion.py))


### 1. Standard deviations

The standard deviations ($\sigma_\mathbf{x}$, i.e. the diagonal elements of the $\mathbf{B}$ matrix) are prescribed based on the absolute value of the prior emissions.

Consider a control variable $x$, controlling for the offsets to the prior emissions for a group of contiguous grid cells $p_i$ to $p_j$, and from time steps $t_n$ to $t_m$. Then the value of $x$ in a given iteration is:

$$ x = \sum_{t = n}^{m}\sum_{p = i}^{j} \left(E(t, p) - E^{apri}(t, p)\right)$$

The prior values of $x$ (i.e. when $E = E^{apri}$) is therefore 0.

The corresponding prior uncertainty, $\sigma_x$, is defined as:

$$\sigma_x = \sum_{t = n}^m\sum_{p = i}^j |E^{apri}(t, p)|$$

In other words, the uncertainty on $x$ is proportional to the sum of the absolute values of the "components" of $x$. This ensures in particular that the uncertainty doesn't drop to zero if the fluxes to which $x$ is an offset happen to cancel out each other (e.g. when respiration and photosynthesis are of equal amplitude).

Note that the absolute values of the standard deviations computed in this step has no importance, as they are scaled to match a target annual uncertainty, in step 3 below. The aim of this step 1 is only to define how this annual uncertainty will be distributed, in time and space.

???+ note "where is it in the code?"
    The standard deviations are directly calculated in the [`lumia.PriorConstraints.setup`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/prior/prior.py) method. It relies on the [`lumia.Mapping.coarsen_cat`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/mapping/multitracer.py) method to aggregate the model-resolution error estimates into a control-vector-like object.

### 2. Correlations

Four correlation methods are implemented, and can be selected using the **cortype** subkey of the **spatial_correlation** and **temporal_correlation** sections. Three of them calculate correlation coefficients based on a mathematical function of the spatial or temporal distance between the grid-cells, whereas the last one reads pre-computed correlation-distance relationships from a file.

???+ note "where is it in the code?"
    The uncertainty settings are read within the [`lumia.Mapping.setup_optimization`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/mapping/multitracer.py) method. In addition, if you need to implement new options, you might need to edit the [`lumia.optimizer.categories.Categories`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/optimizer/categories.py) class.

#### Correlation functions

- Exponential (**cortype** = e): $r = e^{- \left(d / L\right)}$
- Gaussian (**cortype** = g): $r_H = e^{- \left(d / L\right)^2}$
- Hyperbolic (**cortype** = h): $r_H = 1 / \left(1 + d / L\right)$

where $d$ is the geographical distance between the center of two gridcells, and $L$ is a user-specified correlation length (**optimize.emissions.{tracer}.{catname}.horizontal_correlation.cortype** key)

For temporal correlations, only the exponential method is implemented, and **cortype** should be a string understandable by the [`pandas.tseries.frequencies.to_offset`](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html) function.

The last method (**cortype** = f) reads the correlation-distance relationship ($r(d)$) from a file (see below). In which case, a **corrfile** argument is needed (and **corlen** is not used).

???+ note "where is it in the code?"
    The correlation functions are all implemented in the [`lumia.prior.uncertainties`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/prior/uncertainties.py) module (`SpatialCorrelation` and `TemporalCorrelation` classes).

#### Correlation file
<mark>Implementation pending</mark>

It is possible to provide correlation-distance relationships through a file. This allows for more complex correlation functions to be implemented. In order to use this feature, set **cortype** to "f" (or to "file"), and provide the name of the correlation file in a **corfile** key.

The correlation file should be a netCDF file, containing one *correlations* group, with the following structure:

```
group: correlations {
  dimensions:
        point = 1644 ;
        time = 365 ;
  variables:
        double lat(point) ;
                lat:_FillValue = NaN ;
        double lon(point) ;
                lon:_FillValue = NaN ;
        double horizontal_correlation(point, point) ;
                horizontal_correlation:_FillValue = NaN ;
        double temporal_correlations(time, time) ;
                temporal_correlations:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "days since 2018-01-01 00:00:00" ;
                time:calendar = "proleptic_gregorian" ;
        int64 point(point) ;
  } // group correlations
```

The `horizontal_correlations` and `temporal_correlations` variables contain the spatial and temporal matrices. The `time` variable is the coordinate of the `time` dimension of the `temporal_correlations` matrix. The `lat` and `lon` variables store the spatial positions corresponding to the `point` coordinate of the `horizontal_correlation` matrix.

???+ note "where is it in the code?"
    The matrices are read within the `SpatialCorrelation` and `TemporalCorrelation` classes of the [`lumia.prior.uncertainties`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/prior/uncertainties.py) module. Specifically, it uses the `read_spatial_correlations` and `read_temporal_correlations` functions of that module, which doesn't just read the matrices, but also ensure that they are compatible with the current list of coordinates, since there can be *nan* values in the pre-computed matrices.

### 3. Uncertainty scaling

The net uncertainty is calculated as:

$$ \sigma_{tot} = \sqrt{\mathbf{s} \cdot vec(\mathbf{C_h S C_t}^T)} $$

with $s$ the vector containing the standard deviations of $x$, $S$ the same vector, but reshaped as a (*nt*, *np*) matrix (with *nt* and *np* respectively the number of optimization time steps and optimization (clusters of) grid cells), and *vec* the operator reshaping a (*nt*, *np*) matrix as a *np* $\times$ *nt* vector. $\mathbf{C_t}$ and $\mathbf{C_h}$ are the temporal and spatial correlation matrices calculated in step 2.

This relies on the properties that:

- $\mathbf{B = C_t \otimes C_h}$
- the sum of a covariance matrix can be inferred from $\mathbf{s \cdot Q \cdot s^T}$ (with $\mathbf{s}$ the vector of standard deviations, and $\mathbf{Q}$ the correlation matrix
- The equivalence between $(\mathbf{C_t \otimes C_h)} vec(\mathbf{S})$ and $vec(\mathbf{C_h S C_t^T})$

Combining these three equations, we obtain $\mathbf{\Sigma^2} = \mathbf{s \cdot C_t \otimes C_h \cdot s^T = s} \cdot vec(\mathbf{C_h S C_t^T})^T$, with $\mathbf{\Sigma^2}$ the total variance.

Because the formula above doesn't construct the covariance matrix explicitly, but only its sum, it is (relatively) lightweight and efficient to compute. Once the total variance is computed, the standard deviations are multiplied by a scalar factor $\gamma$, defined as

$$ \gamma = \frac{\mathbf{\Sigma_{target}}}{\mathbf{\Sigma}}\frac{\Delta}{\Delta_y} $$

with $\Sigma_{target}$ the target annual uncertainty, $\Delta$ the length of the simulation (in seconds) and $\Delta_y$ the lenght of one year.

???+ note "where is it in the code?"
    The uncertainty scaling is done directly in the [`lumia.PriorConstraints.setup`](https://github.com/lumia-dev/lumia/blob/master/src/lumia/prior/prior.py) method.