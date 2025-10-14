#!/bin/bash

./scripts/cmake_build.sh

HIP_VISIBLE_DEVICES=1 ../build/unittests_app