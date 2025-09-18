GPUS=2

rm -r -f roc*
mkdir rocprof-output
pkill -c 'rocprof'
rocprof-sys-perfetto-traced --background
rocprof-sys-perfetto --out ./rocprof-output/rocprof-sys-perfetto.proto --txt -c /opt/rocprofiler-systems/share/rocprofiler-systems/perfetto.cfg --background
HIP_VISIBLE_DEVICES=$GPUS rocprof-sys-python qwen3_moe_micro_benchmark.py
chmod 777 rocprof-output/rocprof-sys-perfetto.proto 