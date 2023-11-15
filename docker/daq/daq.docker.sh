docker build -f Dockerfile.StreamerDAQ -t streamer-daq:latest --target StreamerDAQ .
docker run --rm -it --name streamer-daq -p 50000-50001:50000-50001 streamer-daq:latest python /Trace/build/python/streamer-daq/DAQStream.py --mode 1 --simulation_file /Trace/data/tomo_00058_all_subsampled1p_s1079s1081.h5 --d_iteration 1 --publisher_addr tcp://*:50000 --iteration_sleep 1 --synch_addr tcp://*:50001 --synch_count 1
