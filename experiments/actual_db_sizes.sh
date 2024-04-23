# sharegpt
for t in 4420 8840 17681 37950 86844 184632 380207 600229 771357 905815; do
    python -m ngram_datastore.build --model-path "lmsys/vicuna-7b-v1.5" --dataset-name "Aeala/ShareGPT_Vicuna_unfiltered" --datastore-path "./datastore/datastore_chat_small.idx" -n4 -c0 -t$t -a
done

# # stack
# for t in 3672 7345 14691 29382 67420 143733 296358 468061 601608 706538; do
#     python -m ngram_datastore.build --model-path "codellama/CodeLlama-7b-instruct-hf" --dataset-name "bigcode/the-stack-dedup" --datastore-path "./datastore/datastore_stack_small.idx" -n5 -c0 -t$t -a
# done
