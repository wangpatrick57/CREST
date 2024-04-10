#!/bin/bash
ngram_ns=(2 3 4 5 6)
merge_ratios=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 0)

for ngram_n in "${ngram_ns[@]}"; do
    for merge_ratio in "${merge_ratios[@]}"; do
        RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 -m llm_judge.gen_virtual_ngram_model_answer_rest --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --datastore-path datastore/datastore_chat_small.idx -n$ngram_n -t0 -m$merge_ratio -o"accept_length_n${ngram_n}_m${merge_ratio}.out"
    done
done
