FROM rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

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