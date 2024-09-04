# Simplified Mini-App for Batch Execution without Streaming

## Requirements

1. **HDF5**
    On Polaris
    ```bash
    $ module load cray-hdf5-parallel
    ```

2. **VeloC**
    Source code and installation available at [VeloC]([gitter](https://veloc.readthedocs.io/en/latest/)

## Build on Polaris
    ```bash
    $ ./build.sh
    ```


## To reconstruct
    ```bash
    $ ./art_simple_main ../../../data/tooth_preprocessed.h5 294.078 <# outer iterations> <# inner iterations> ../art_simple.cfg
    # For examples
    ./art_simple_main ../../../data/tooth_preprocessed.h5 294.078 5 2 ../art_simple.cfg
    ```


