#!/bin/bash


run_recon() {
	nprocs=$1
	nslices=$2
	skip_ratio=$3
	skip_begin=$4
	write_to_file=$5
	checkpoint=$6
	
	rm -rf /tmp/scratch
	rm -rf /tmp/persistent

	mpiexec -np $nprocs ./art_simple_ckpt_quality_analysis ../../../data/tooth_preprocessed.h5 294.078 10 1 \
		0 $nslices \
		$skip_ratio $skip_begin \
		$write_to_file \
		$checkpoint ../art_simple.cfg
}

fix_compute_vary_data() {
	nslices=$1
	end_slices=$2
	while [ $nslices -le $end_slices ]; do
        echo "RUN WITH NSLICES = $nslices -------------------------------------------------------------------------------------- "
		echo "CHPT VELOC ==============================================================================="
		run_recon 1 $nslices 0 0 0 1
		echo "CKPT BLOCKING ============================================================================"
		run_recon 1 $nslices 0 0 1 0
		echo "CKPT NONE ================================================================================"
		run_recon 1 $nslices 0 0 0 0
		nslices=$((nslices * 2))
	done
}

fix_data_vary_compute() {
	nslices=$1
	nprocs=1
	while [ $nprocs -le $nslices ]; do
		echo "RUN WITH NPROCS = $nprocs / $nslices --------------------------------------------------------------------------- "
		echo "CHPT VELOC ==============================================================================="
		run_recon $nprocs $nslices 0 0 0 1
		echo "CKPT BLOCKING ============================================================================"
		run_recon $nprocs $nslices 0 0 1 0
		echo "CKPT NONE ================================================================================"
		run_recon $nprocs $nslices 0 0 0 0
		nprocs=$((nprocs * 2))
	done
}

#fix_compute_vary_data 1 64
fix_data_vary_compute 1



