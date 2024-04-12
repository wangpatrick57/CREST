#!/bin/bash

# Array of values for i
# values=(3 4 5 6 7 8 9 10 12 14 16 18 20 30 40 50 60 70 80 90)
# # values=(2)

# # Loop through the values of i
# for i in "${values[@]}"; do
    # Construct the datastore file name

    # file_name="datastore_chat_small_${i}_percent.idx"
    
    # Run the command with the current file in the background

echo "Started running experiment for datastore: datastore_chat_small_1_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_1_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_2_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_2_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_3_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_3_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_4_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_4_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_5_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_5_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_6_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_6_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_7_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_7_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_8_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_8_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_9_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_9_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_10_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_10_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_12_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_12_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_14_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_14_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_16_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_16_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_18_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_18_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_20_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_20_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_30_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_30_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_40_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_40_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_50_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_50_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_60_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_60_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_70_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_70_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_80_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_80_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_90_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_90_percent.idx"

echo "Started running experiment for datastore: datastore_chat_small_100_percent"

RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python3 gen_model_answer_rest.py \
--model-path lmsys/vicuna-7b-v1.5 \
--model-id vicuna-7b-v1.5 \
--answer-file "data/mt-bench/model_answer/rest_experiment_all.jsonl" \
--datastore-path "../datastore/datastore_chat_small_100_percent.idx"

# Finshed all the experiments
echo "All experiments completed"