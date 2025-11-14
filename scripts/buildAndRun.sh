#!/bin/bash

./scripts/cmake_build.sh

HIP_VISIBLE_DEVICES=0 ../build/unittests_app --gtest_filter="*MoeExperts*"


# qwen3 experts = 128!