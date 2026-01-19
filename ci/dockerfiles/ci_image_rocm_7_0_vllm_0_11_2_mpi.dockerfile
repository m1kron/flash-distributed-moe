FROM rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210

RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        locales-all \
        make \
        python3-pip \
        ssh \
        sudo \
        wget \
        cmake \
        gdb \
        vim 

RUN git clone https://github.com/ROCm/ucx.git -b v1.17.x
RUN cd ucx
RUN ./autogen.sh
RUN ./configure --prefix=/opt/mpi/ucx --with-rocm=/opt/rocm --enable-mt
RUN make -j 8
RUN make -j 8 install
RUN cd ..
RUN rm -rf ucx

RUN git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
RUN cd ompi
RUN ./autogen.pl
RUN ./configure --prefix=/opt/mpi/ompi --with-rocm=/opt/rocm --with-ucx=/opt/mpi/ucx
RUN make -j 8
RUN make -j 8 install
RUN cd ..
RUN rm -rf ompi

RUN echo "export LD_LIBRARY_PATH=/opt/rocprofiler-systems/lib/rocprofiler-systems/:/usr/local/lib/python3.12/dist-packages/torch/lib/:/opt/mpi/ompi/lib" >> ~/.bashrc