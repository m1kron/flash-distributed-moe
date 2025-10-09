#!/bin/bash

./scripts/cmake_build.sh

CUDA_VISIBLE_DEVICES=1 ../build/unittests_app