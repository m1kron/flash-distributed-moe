GPU=2

rm -r -f roc*
mkdir rocprof-output
pkill -c 'rocprof'
rocprof-sys-perfetto-traced --background
rocprof-sys-perfetto --out ./rocprof-output/rocprof-sys-perfetto.proto --txt -c /opt/rocprofiler-systems/share/rocprofiler-systems/perfetto.cfg --background
HIP_VISIBLE_DEVICES=$GPU PYTHONPATH=../build rocprof-sys-python test/python/qwen3MoeMicroBenchmark.py 
chmod 777 rocprof-output/rocprof-sys-perfetto.proto 