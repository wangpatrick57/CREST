#!/bin/bash

# echo "Started running experiment for datastore: datastore_chat_large_1_percent"

# RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
# --model-path lmsys/vicuna-7b-v1.5 \
# --model-id vicuna-7b-v1.5 \
# --answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
# --datastore-path "../datastore/datastore_chat_large_1_percent.idx"

# echo "Started running experiment for datastore: datastore_chat_large_2_percent"

# RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
# --model-path lmsys/vicuna-7b-v1.5 \
# --model-id vicuna-7b-v1.5 \
# --answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
# --datastore-path "../datastore/datastore_chat_large_2_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_4_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large_4_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_8_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large_8_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_16_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large_16_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_32_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large_32_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_64_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large_64_percent.idx"

echo "Started running experiment for datastore: datastore_chat_large_100_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_large.idx"

# Finshed all the experiments
echo "All experiments completed"