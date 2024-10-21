#!/usr/bin/env bash
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set -eux

# ensure this script is in the cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

output_dir=hiptensor-benchmarks
build_dir=../../build/bin/
config_dir=../../test/03_reduction/configs/bench

cold_runs=1
hot_runs=5

validate=OFF

if [ -d "$build_dir" ]; then
    # setup output directory for benchmarks
    mkdir -p "$output_dir"

    tests=("rank2_reduction_test"
           "rank3_reduction_test"
           "rank4_reduction_test"
           "rank5_reduction_test"
           "rank6_reduction_test")

    configs=("rank2_test_params.yaml"
             "rank3_test_params.yaml"
             "rank4_test_params.yaml"
             "rank5_test_params.yaml"
             "rank6_test_params.yaml")

    arrayLength=${#tests[@]}

    # run benchmarks
    for (( i=0; i<${arrayLength}; i++ )); do
        if [[ -e $build_dir && ! -L $build_dir/${tests[$i]} ]]; then
            $build_dir${tests[$i]} -y $config_dir/${configs[$i]} \
            -o $output_dir${tests[$i]}".csv" --cold_runs $cold_runs --hot_runs $hot_runs -v $validate
        fi
    done
fi

