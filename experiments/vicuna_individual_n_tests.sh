#!/bin/bash
NUM_BENCHMARK_CONVS=20

# # Use this for testing since it completes the fastest
# RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 -m llm_judge.gen_virtual_ngram_model_answer_rest --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --datastore-path datastore/datastore_chat_small.idx -n2 -t0 -m1.0 -b1 -o"experiments/vicuna_individual_n_tests_outputs/accept_length_n2_m1.0-b1.out"
# exit 0

ngram_ns=(1 2 3 4 5)
merge_ratios=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.0)

for ngram_n in "${ngram_ns[@]}"; do
    for merge_ratio in "${merge_ratios[@]}"; do
        RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 -m llm_judge.gen_virtual_ngram_model_answer_rest --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --datastore-path datastore/datastore_chat_small.idx -n$ngram_n -t0 -m$merge_ratio -b$NUM_BENCHMARK_CONVS -o"experiments/vicuna_individual_n_tests_outputs/accept_length_n${ngram_n}_m${merge_ratio}-b${NUM_BENCHMARK_CONVS}.out"
    done
done
