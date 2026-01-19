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
    make -j 32 && \
    make install && \
    cd .. && \
    rm -rf ucx

RUN git clone --recursive https://github.com/ROCm/ompi.git && \
    cd ompi && git checkout 697a596dde68815fe50db3c2a75a42ddb41b5ef4 && \
    ./autogen.pl && \
    ./configure --prefix=/opt/mpi/ompi --with-rocm=/opt/rocm --with-ucx=/opt/mpi/ucx --disable-oshmem --with-prrte=internal --with-hwloc=internal --with-libevent=internal --without-cuda --disable-mpi-fortran --without-ofi && \
    make -j 32 && \
    make install && \
    cd .. && \
    rm -rf ompi

WORKDIR /

ENV LD_LIBRARY_PATH="/opt/mpi/ompi/lib/openmpi:$LD_LIBRARY_PATH"

RUN ls /opt/mpi/ompi/