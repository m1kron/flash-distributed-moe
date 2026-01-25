GPUS=2

export ROCPROFSYS_CONFIG_FILE=~/.rocprof-sys.cfg

rm -r -f rocp*
mkdir rocprof-output
pkill -c 'rocprof'
rocprof-sys-perfetto-traced --background
rocprof-sys-perfetto --out ./rocprof-output/rocprof-sys-perfetto.proto --txt -c /opt/rocprofiler-systems/share/rocprofiler-systems/perfetto.cfg --background

./test/multiGPU/python/start.sh

chmod 777 rocprof-output/rocprof-sys-perfetto.proto 