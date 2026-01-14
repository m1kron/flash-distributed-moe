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