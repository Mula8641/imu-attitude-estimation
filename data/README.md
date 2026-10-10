# Data

No data is committed. The benchmark uses BROAD, the Berlin Robust Orientation Estimation Assessment Dataset.

## Download
```bash
git clone --depth 1 https://github.com/dlaidig/broad.git broad_clone
mkdir -p data/broad && cp broad_clone/data_hdf5/*.hdf5 data/broad/
```
(about 175 MB of HDF5 files are needed; the full repository is larger.)

## Layout
`data/broad/NN_<undisturbed|disturbed>_<description>.hdf5`, with keys `imu_acc`, `imu_gyr`, `opt_quat`, `movement` and the attribute `sampling_rate`. See `docs/environment.md`.

## Credit
BROAD is licensed under CC BY 4.0. D. Laidig, M. Caruso, A. Cereatti, T. Seel, "BROAD - A Benchmark for Robust Inertial Orientation Estimation", Data 2021, 6(7), 72. https://doi.org/10.3390/data6070072
