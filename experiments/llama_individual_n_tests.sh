#!/bin/bash
NUM_BENCHMARK_CONVS=1

# # Use this for testing since it completes the fastest
# RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 -m human_eval.virtual_rest_test --model-path codellama/CodeLlama-7b-instruct-hf --datastore-path ./datastore/datastore_stack_small.idx -n1 -t0 -m1.0 -b1 -o"experiments/llama_individual_n_tests_outputs/accept_length_n2_m1.0-b1.out"
# exit 0

ngram_ns=(1 2 3 4 5)
merge_ratios=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.0)

for ngram_n in "${ngram_ns[@]}"; do
    for merge_ratio in "${merge_ratios[@]}"; do
        RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 -m human_eval.virtual_rest_test --model-path codellama/CodeLlama-7b-instruct-hf --datastore-path ./datastore/datastore_stack_small.idx -n$ngram_n -t0 -m$merge_ratio -b$NUM_BENCHMARK_CONVS -o"experiments/llama_individual_n_tests_outputs/accept_length_n${ngram_n}_m${merge_ratio}-b${NUM_BENCHMARK_CONVS}.out"
    done
done
