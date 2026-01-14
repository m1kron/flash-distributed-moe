GPUS=2

export ROCPROFSYS_CONFIG_FILE=~/.rocprof-sys.cfg

rm -r -f rocp*
mkdir rocprof-output
pkill -c 'rocprof'
rocprof-sys-perfetto-traced --background
rocprof-sys-perfetto --out ./rocprof-output/rocprof-sys-perfetto.proto --txt -c /opt/rocprofiler-systems/share/rocprofiler-systems/perfetto.cfg --background

HIP_VISIBLE_DEVICES=4,5,6,7 python3 benchmark_dp.py -dp=4 --all2all-backend=allgather_reducescatter --disable-nccl-for-dp-synchronization

chmod 777 rocprof-output/rocprof-sys-perfetto.proto 