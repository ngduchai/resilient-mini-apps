#!/bin/bash -l
#PBS -N AFFINITY
#PBS -l select=4:ncpus=256
#PBS -l walltime=0:10:00
#PBS -q debug
#PBS -A diaspora

#NNODES=`wc -l < $PBS_NODEFILE`
NNODES=1
NRANKS=2 # Number of MPI ranks to spawn per node
NDEPTH=8 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=8 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

# Change the directory to work directory, which is the directory you submit the job.
PBS_O_WORKDIR=/home/ndhai/diaspora/src/resilient-mini-apps/src/art_simple/build
cd $PBS_O_WORKDIR
mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -env OMP_NUM_THREADS=${NTHREADS} ./art_simple_main /eagle/Diaspora/bicer/data/tomobank/shale/tomo_00001/tomo_00001_normalized_mlogged_striperemoved.h5 1024 5 2 520 2 ../art_simple.cfg


