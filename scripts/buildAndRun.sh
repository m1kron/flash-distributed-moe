#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

./scripts/cmake_build.sh

HIP_VISIBLE_DEVICES=1 ../build/test/cpp/unittests_app --gtest_filter="*MoeFull*"