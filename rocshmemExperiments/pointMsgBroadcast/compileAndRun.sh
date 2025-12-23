hipcc -c -fgpu-rdc -x hip pointMsgBroadcast.cpp \
  --offload-arch=gfx942:xnack- \
  -I/opt/rocm/include \
  -I/opt/mpi/install/ompi/include/

hipcc -fgpu-rdc --hip-link pointMsgBroadcast.o -o pointMsgBroadcastApp \
  --offload-arch=gfx942:xnack- \
  /opt/rocm/lib/librocshmem.a \
  /opt/mpi/install/ompi/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

rm pointMsgBroadcast.o 

ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun --allow-run-as-root -np 8 ./pointMsgBroadcastApp