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
        vim \
        autoconf \
        automake \
        libtool \
        texinfo \
        flex

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

# From rocshmem/install_dependencies.sh
WORKDIR /tmp
RUN git clone https://github.com/ROCm/ucx.git && \
    cd ucx && \
    git checkout 18770fdc1c3b5de202d14a088a14b734d2c4bbf3 && \
    ./autogen.sh && \
    ./contrib/configure-release --prefix=/opt/mpi/ucx --with-rocm=/opt/rocm --enable-mt --without-go --without-java --without-cuda --without-knem && \
    make -j && \
    make install && \
    cd .. && \
    rm -rf ucx

RUN git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x && \
    cd ompi && \
    ./autogen.pl && \
    ./configure --prefix=/opt/mpi/ompi --with-rocm=/opt/rocm --with-ucx=/opt/mpi/ucx && \
    make -j 8 && \
    make install && \
    cd .. && \
    rm -rf ompi

WORKDIR /

RUN echo "export LD_LIBRARY_PATH=/opt/rocprofiler-systems/lib/rocprofiler-systems/:/usr/local/lib/python3.12/dist-packages/torch/lib/:/opt/mpi/ompi/lib" >> ~/.bashrc