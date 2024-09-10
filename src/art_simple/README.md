# Simplified Mini-App for Batch Execution without Streaming

## Requirements

1. **HDF5**

   On Polaris
    ```bash
    $ module load cray-hdf5-parallel

3. **VeloC**

   Source code and installation available at [VeloC](https://veloc.readthedocs.io/en/latest/)

## Build

On Polaris

```bash
$ ./build.sh
```

## To reconstruct

Go to the build directory

```bash
cd build
```

then try

```bash
$ mpiexec -np <# processes> ./art_simple_main ../../../data/tooth_preprocessed.h5 294.078 <# outer iterations> <# inner iterations> ../art_simple.cfg
```
    
For examples

```bash
$ mpiexec -np 2 ./art_simple_main ../../../data/tooth_preprocessed.h5 294.078 5 2 ../art_simple.cfg
```


