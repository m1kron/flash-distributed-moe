#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Build cpp project
./scripts/cmake_build.sh

# Build python torch binding
python3 python/setup.py build_ext

# Run cpp tests
HIP_VISIBLE_DEVICES=1 ../build/test/cpp/unittests_app

# Run pytest
HIP_VISIBLE_DEVICES=1 PYTHONPATH=/home/REPO/flash-moe/build pytest