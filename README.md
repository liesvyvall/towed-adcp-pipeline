# towed-adcp-pipeline

Processing of underway, bottom-tracking ADCP data collected from a moving boat: from the ASCII files
exported by WinRiver to gridded fields of tidal and residual flow. The code follows the sequence of
matrices described by Valle-Levinson (2024): **bronze**, **silver**, **gold** and **platinum**.

The chapter provides Matlab scripts (`bronze.m`, `silver.m`, `gold.m`, `platinum.m`, `lsqfit.m`) and
an example data set for a single straight transect; they are distributed as supplementary material
of the chapter on the publisher's page (Appendix A). This package does the same thing in Python and
adds two things I needed for my own surveys: a way to split a closed circuit that is sailed over and
over into several transects, and a rotation of the results to the channel axis.

## What it does

A towed ADCP survey repeats the same track N times during one or more tidal cycles. Each pass over
the track is a *repetition*. The processing turns the raw profiles into, at every point of a
regular grid on the transect, a residual flow plus the amplitude and phase of a few tidal bands.

| Step | Matrix | What happens |
|---|---|---|
| 1-2 | (WinRiver) | Average the raw pings and export to the classic ASCII format. Not done here. |
| 3 | bronze | Read the ASCII files, keep the usable bins and stack everything in one table. |
| 4 | silver | Calibrate the compass for each repetition (Joyce, 1989) and correct *u*, *v*. |
| 5 | gold | Interpolate each repetition onto a regular (distance, depth) grid. |
| 6 | platinum | Least-squares fit at every grid node: residual + harmonic bands. |

Between silver and gold you need to know when each repetition starts and ends. There are two ways
to get that (see below).

### Bronze

One row per bin, 11 columns: time, latitude, longitude, *u*, *v*, bin depth, water-column depth,
bottom-track *u* and *v*, mean backscatter of the four beams and transducer temperature. A bin is
kept when |error velocity| < 10 cm/s and |discharge| < 100; the percent-good test (> 70 in the
chapter) is optional because some exports report 0 in every bin. A whole profile is dropped when
there is no bottom track or the boat was moving slower than 0.1 m/s.

Time is the decimal day of the month (19.5 is day 19 at 12:00), the same convention the chapter
uses, so the `timtran` files of the chapter can be used directly. The origin (year, month) is
stored with the matrix so it can be converted back to dates.

### Silver

The water velocity reported by the ADCP is the measured velocity minus the bottom-track velocity,
and the bottom track is referred to the instrument compass. For each repetition, the bottom track
(*u_bt*, *v_bt*) is compared with the boat velocity from consecutive GPS fixes (*u_sh*, *v_sh*) to
get a rotation α and a scaling 1 + β:

    tan α   = <u_bt v_sh - v_bt u_sh> / <u_bt u_sh + v_bt v_sh>
    1 + β   = sqrt( <u_sh² + v_sh²> / <u_bt² + v_bt²> )

    u_c = (1 + β) (u cos α - v sin α)
    v_c = (1 + β) (u sin α + v cos α)

Normal values are |α| < 0.2 rad and |β| < 0.03. The silver matrix has 9 columns: the first seven
of the bronze matrix with the corrected velocities, backscatter and temperature. Since the bias
depends on the heading, the correction is computed per transect when the circuit has several legs
with different headings.

### Gold

For every transect and repetition, the triads (distance, bin depth, value) are interpolated with
linear Delaunay triangulation onto a fixed grid (`scipy.interpolate.griddata`, the same method as
Matlab's `griddata`). The distance is either the projection of each profile on the axis that joins
the two end points of the transect, or the distance to an origin as in the chapter's `gold.m`. Grid
spacing should not be finer than what was measured: Δy ≥ Δt × boat speed, Δz ≥ bin size. Nodes
below the bed of that repetition are set to NaN.

The result for each transect is a set of arrays of shape (depth × distance × repetition) for
*u_c*, *v_c*, time and backscatter. Every grid node now holds a time series with one value per
repetition.

### Platinum

At every node the series is fitted to

    u(t) = u_r + Σ_i u_0i sin(ω_i t + φ_i)

with the frequencies ω_i known (for example diurnal, semidiurnal and quarter-diurnal bands). With
the identity sin(A + B) the problem is linear and the normal equations are the system **B** =
**A X** of the chapter; here **A** = **G**ᵀ**G** and **B** = **G**ᵀ**u**, with **G** the matrix
whose columns are [1, sin ω₁t, cos ω₁t, sin ω₂t, ...]. The platinum matrix has shape (depth ×
distance × 2M + 1): the residual, then amplitude and phase of each band. R², Willmott skill and
RMSE are stored for every node.

Optionally the gold matrices are rotated to the channel axis (the principal axis of the
depth-averaged current, or a heading you give) and fitted again, which gives the along-channel
and cross-channel platinum directly.

## Defining the repetitions

`towed_adcp.transects` offers two methods. Both return the same table with columns
`lap, transect, t_start, t_end, dur_min, d_start, d_end, ok`.

**From a file** (`repetitions_from_file`). Read the start and end times from a file, like the
chapter's `timtran1.mat`. Accepted formats: a `.mat` with one vector alternating start, end, start,
end, ...; a text file with one column in the same order or with two columns (start, end); or a CSV
with a header and the columns `transect, t_start, t_end`. Times must be in the same unit as the
bronze matrix (decimal day of the month). This is the way to go for a single line sailed back and
forth, where you pick the times by hand from the track.

**From the track** (`repetitions_by_vertices`). When the survey is a closed circuit that is repeated
the same way every time, give the coordinates of its turning points (the vertices) and the order in
which they are visited. The code computes the distance from every profile to every vertex; each
local minimum of the distance to the first vertex starts a new lap, and within a lap the times of
closest approach to the other vertices, in order, bound the transects. A repetition is accepted
when the boat came within `r_max` of both end points, took less than `dur_max` minutes and has no
gaps longer than `gap_max` seconds (file changes). To pick the vertex coordinates, plot one lap,
find the turns, and take the median position of each turn over all laps.

## Installation

```
git clone https://github.com/liesvyvall/towed-adcp-pipeline.git
cd towed-adcp-pipeline
pip install -e .
```

Requirements: numpy, scipy, pandas, matplotlib, cmocean and pyyaml. Maps with a coastline need
geopandas and shapely (`pip install -e ".[maps]"`).

## Usage

### Command line

Write a YAML configuration (two examples in `examples/`) and run

```
python scripts/run_pipeline.py examples/config_closed_circuit.yml
```

The outputs go to the `output` folder named in the config: `bronze.npz`, `silver.npz`,
`repetitions.csv` (with α and β of every repetition), `gold.npz`, `platinum.npz`, and, if the
rotation is on, `platinum_channel.npz` and `channel_axis.txt`, plus a `figs/` folder with the
figures. Data files are never part of this repository; the examples expect them in a `data/` folder
that git ignores.

`examples/config_closed_circuit.yml` is a figure-eight circuit through the mouth of a small estuary,
sailed about 50 times in 25 hours and split into six transects between six turning points.
`examples/config_single_line.yml` is one straight transect with the repetition times read from a
`.mat` file, the layout of the example that comes with the chapter.

### Python

```python
import towed_adcp as ta
from towed_adcp import plotting as pl

bronze, meta = ta.build_bronze('data/circuit/*_ASC.TXT', pg_min=None)
profiles = ta.profiles_from_bronze(bronze)

vertices = {'SW': (-116.62278, 31.76041), 'SE': (-116.62034, 31.76074), 'NW': (-116.62124, 31.76778),
            'NE': (-116.61951, 31.76699), 'CW': (-116.62245, 31.76505), 'CE': (-116.62008, 31.76436)}
transects = {'South': ['SW', 'SE'], 'South_North': ['SE', 'NW'], 'North': ['NW', 'NE'],
             'North_Center': ['NE', 'CW'], 'Center': ['CW', 'CE'], 'Center_South': ['CE', 'SW']}
reps = ta.repetitions_by_vertices(profiles, vertices, transects)
# or: reps = ta.repetitions_from_file('data/line/timtran.mat', transect='line')

silver, reps = ta.build_silver(bronze, reps)
geometry = ta.transect_geometry(transects, vertices)
gold = ta.build_gold(silver, reps, geometry, dy=10, dz=0.25, z_min=0.5, z_max=7)
platinum = ta.build_platinum(gold, {'D1': 23.93, 'D2': 12.42, 'D4': 6.21}, n_min=25)

theta, frac = ta.principal_axis(silver)          # channel axis
gold_ch = ta.rotate_gold(gold, theta)            # 'u' along channel, 'v' across
platinum_ch = ta.build_platinum(gold_ch, {'D1': 23.93, 'D2': 12.42, 'D4': 6.21}, n_min=25)

pl.set_style()
pl.plot_platinum_panel(gold['Center'], platinum['Center'], 'Center', meta['time_base'])
pl.plot_residual_sections(gold_ch, platinum_ch, 'u', 'Along-channel residual (cm/s)')
```

`examples/walkthrough.ipynb` goes through the same steps one at a time with the figures.

### Figures

`towed_adcp.plotting` has one function per figure: the ship track, the bronze time series and
histograms, the repetitions on the map, α and β, the velocities before and after the correction,
panels of gold repetitions, the fit at one node with an increasing number of bands, the platinum
panels (residual, amplitudes, hour of the maximum of the main band, R²), the residual and tidal
ellipses on the map, the along- and cross-channel residual, and maps of the transducer temperature
averaged over a few hours.

## Input format

The reader expects the classic ASCII output of WinRiver II: a first line with the configuration
(bin size, blank, number of bins, pings per ensemble, ...), and for every profile six header lines
(date and time, ensemble number, pitch, roll, heading, temperature; bottom-track velocities and
depth by beam; distances made good; latitude, longitude and GPS velocity; discharges; number of
bins and units) followed by one row per bin with depth, magnitude, direction, east, north, vertical
and error velocities, echo intensity of the four beams, percent good and discharge. Bad values are
flagged with −32768. If your export has a different layout, `towed_adcp.io.read_winriver_ascii` is
the only function to adapt.

## References

Valle-Levinson, A. (2024). Collection and processing of underway, bottom-tracking ADCP data. In:
Baird, D. and Elliott, M. (eds.), *Treatise on Estuarine and Coastal Science*, 2nd edition, vol. 2,
pp. 207–218. Elsevier. https://doi.org/10.1016/B978-0-323-90798-9.00023-8

Joyce, T. M. (1989). On in situ "calibration" of shipboard ADCPs. *Journal of Atmospheric and
Oceanic Technology*, 6, 169–172.

Pollard, R. T. and Read, J. (1989). A method for calibrating shipmounted acoustic Doppler current
profilers and the limitations of gyro compasses. *Journal of Atmospheric and Oceanic Technology*,
6, 859–865.

Trump, C. L. and Marmorino, G. (1997). Calibrating a gyrocompass using ADCP and DGPS data.
*Journal of Atmospheric and Oceanic Technology*, 14, 211–214.

Fang, T. and Piegl, L. A. (1993). Delaunay triangulation using a uniform grid. *IEEE Computer
Graphics and Applications*, 13, 36–47.

Willmott, C. J. (1981). On the validation of models. *Physical Geography*, 2, 184–194.

## How to cite

If this code is useful in your work, please cite both the software and the chapter that describes
the method:

> Valladares, L. (2026). towed-adcp-pipeline: bronze, silver, gold and platinum matrices from
> underway, bottom-tracking ADCP data (version 0.1.0). https://github.com/liesvyvall/towed-adcp-pipeline

> Valle-Levinson, A. (2024). Collection and processing of underway, bottom-tracking ADCP data. In:
> Baird, D. and Elliott, M. (eds.), *Treatise on Estuarine and Coastal Science*, 2nd edition, vol. 2,
> pp. 207–218. Elsevier. https://doi.org/10.1016/B978-0-323-90798-9.00023-8

A `CITATION.cff` file with both entries is included for reference managers.

## License

MIT. You can use, modify and redistribute the code, provided the copyright notice and the license
text stay with it. See `LICENSE`.
