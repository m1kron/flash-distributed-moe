#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Build cpp project
./scripts/cmake_build.sh

# Build python torch binding
python3 python/setup.py build_ext

# Run cpp tests
HIP_VISIBLE_DEVICES=0 ../build/test/singleGPU/cpp/singleGPU_unittests_app

# Run pytest
HIP_VISIBLE_DEVICES=0 PYTHONPATH=../build pytest