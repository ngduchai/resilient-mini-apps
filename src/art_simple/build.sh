#module use /soft/modulefiles
#module load spack-pe-base cmake
#module load cray-hdf5-parallel
mkdir build; cd build; cmake ..; make; mkdir tmp; cd ..;
cp submit-art-simple.sh build

