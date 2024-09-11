module load cray-hdf5-parallel
module use /soft/modulefiles
module load spack-pe-base cmake
mkdir build; cd build; cmake ..; make; mkdir tmp; cd ..;
cp job.sh build/job.sh

