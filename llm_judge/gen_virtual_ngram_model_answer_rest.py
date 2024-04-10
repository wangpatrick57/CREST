"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import pickle
import random
import time
import shortuuid
import torch
import numpy as np
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import load_model, get_conversation_template

# Rest imports
import transformers

import sys

from ngram_datastore.ngram_datastore import NGramDatastoreBuilder
from ngram_datastore.utils import get_ngrams_from_sharegpt
sys.path.append("../")

from ngram_datastore.ngram_datastore_settings import NGramDatastoreSettings
from rest.model.utils import *
from rest.model.rest_model import RestModel
from rest.model.kv_cache import initialize_past_key_values
import draftretriever


def virtual_generate_candidates_and_draft_buffer(logits, input_ids, datastore, token_spans, filtered_ngrams, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()

        if tuple(this_token) in filtered_ngrams:
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
    
        # No retrieved sequences
        if len(retrieved_token_list) == 0:
            continue
        # Break because this span has hitted
        else:
            break
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


def rest_forward(input_ids, model, tokenizer, max_new_token, temperature, top_p, datastore, num_draft, token_spans, filtered_ngrams, max_steps=1024):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len + 1
    model.base_model.model.draft_mask = None
    logits = initialize_logits(
            input_ids, model, past_key_values
    )
    new_token = 0
    
    torch.cuda.synchronize()
    start_time = time.time()
    for idx in tqdm(range(max_steps)): 
        candidates, tree_candidates, draft_buffers = virtual_generate_candidates_and_draft_buffer(
                logits,
                input_ids,
                datastore,
                token_spans,
                filtered_ngrams,
                top_p=top_p,
                temperature=temperature,
                max_num_draft=num_draft,
                device=model.base_model.device,
            )
        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
        logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                draft_buffers["draft_position_ids"],
                input_ids,
                draft_buffers["retrieve_indices"],
            )
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, top_p
            )
        input_ids, logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                draft_buffers["retrieve_indices"],
                outputs,
                logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_token:
            break
    return input_ids, new_token, idx, accept_length_list, start_time

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    top_p,
    datastore_path,
    num_draft,
    max_token_span,
    ngram_datastore_settings: NGramDatastoreSettings,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    token_spans = list(range(1, max_token_span+1))[::-1]
    print("loading the datastore ...")
    datastore = draftretriever.Reader(
                index_file_path=datastore_path,
            )
    print("datastore loaded!")
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    for i in tqdm(range(0, len(questions), chunk_size)):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                top_p,
                datastore,
                num_draft,
                token_spans,
                ngram_datastore_settings,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def fast_get_sorted_ngrams(dataset_name, ngram_n):
    with open(f"llm_judge/{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-{ngram_n}gram-set.pkl", "rb") as file:
        return pickle.load(file)


def get_filtered_ngrams(settings: NGramDatastoreSettings, tokenizer):
    filtered_ngrams = set()
    ngram_ns_to_include = list(range(1, settings.ngram_n + 1)) if settings.include_all else [settings.ngram_n]

    for ngram_n in ngram_ns_to_include:
        sorted_ngrams = fast_get_sorted_ngrams(settings.dataset_name, settings.ngram_n)

        if settings.merge_ratio != 0.0:
            top_ngrams = sorted_ngrams[:int(len(sorted_ngrams) * settings.merge_ratio)]
        elif settings.num_top_ngrams != 0:
            top_ngrams = sorted_ngrams[:settings.num_top_ngrams]
        else:
            top_ngrams = sorted_ngrams
        
        for top_ngram in top_ngrams:
            filtered_ngrams.add(top_ngram)
    
    return filtered_ngrams


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    top_p,
    datastore,
    num_draft,
    token_spans,
    ngram_datastore_settings: NGramDatastoreSettings,
):
    
    model = RestModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    
    question = questions[0]

    filtered_ngrams = get_filtered_ngrams(ngram_datastore_settings, tokenizer)

    # warmup
    # for _ in range(3):
    #     torch.manual_seed(0)
    #     conv = get_conversation_template(model_id)
    #     turns = []
    #     idxs = []
    #     new_tokens = []
    #     wall_time = []
    #     for j in range(len(question["turns"])):
    #         qs = question["turns"][j]
    #         conv.append_message(conv.roles[0], qs)
    #         conv.append_message(conv.roles[1], None)
    #         prompt = conv.get_prompt()
    #         input_ids = tokenizer([prompt]).input_ids

    #         # if temperature < 1e-4:
    #         #     do_sample = False
    #         # else:
    #         #     do_sample = True

    #         # some models may error out when generating long outputs
    #         try:
    #             print("before the forward pass")
    #             output_ids, new_token, idx, _, start_time = rest_forward(
    #                 torch.as_tensor(input_ids).cuda(),
    #                 model,
    #                 tokenizer,
    #                 max_new_token,
    #                 temperature,
    #                 top_p,
    #                 datastore,
    #                 num_draft,
    #                 token_spans,
    #             )
    #             print("finished the forward pass")
    #             torch.cuda.synchronize()
    #             total_time = time.time() - start_time
    #             output_ids = output_ids[0][len(input_ids[0]) :]
    #             # be consistent with the template's stop_token_ids
    #             if conv.stop_token_ids:
    #                 stop_token_ids_index = [
    #                     i
    #                     for i, id in enumerate(output_ids)
    #                     if id in conv.stop_token_ids
    #                 ]
    #                 if len(stop_token_ids_index) > 0:
    #                     output_ids = output_ids[: stop_token_ids_index[0]]

    #             print("starting the tokenizer decode")
    #             output = tokenizer.decode(
    #                 output_ids,
    #                 spaces_between_special_tokens=False,
    #             )
    #             if conv.stop_str and output.find(conv.stop_str) > 0:
    #                 output = output[: output.find(conv.stop_str)]
    #             # for special_token in tokenizer.special_tokens_map.values():
    #             #     if isinstance(special_token, list):
    #             #         for special_tok in special_token:
    #             #             output = output.replace(special_tok, "")
    #             #     else:
    #             #         output = output.replace(special_token, "")
    #             # output = output.strip()

    #             if conv.name == "xgen" and output.startswith("Assistant:"):
    #                 output = output.replace("Assistant:", "", 1).strip()
    #         except RuntimeError as e:
    #             print(f"question ID {question['question_id']} errored out with {e}")
    #             output = "ERROR"

    #         turns.append(output)
    #         idxs.append(int(idx))
    #         new_tokens.append(int(new_token))
    #         wall_time.append(total_time)
    #         conv.messages[-1][-1] = output
    print('Skipping warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions[:1]):
        # if question["category"] in temperature_config:
        #     temperature = temperature_config[question["category"]]
        # else:
        #     temperature = 0.7
        choices = []
        # for i in range(num_choices):
        for i in range(1):
            accept_lengths_tree_this = []
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # if temperature < 1e-4:
                #     do_sample = False
                # else:
                #     do_sample = True

                # some models may error out when generating long outputs
                try:

                    output_ids, new_token, idx, accept_length_tree, start_time = rest_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        max_new_token,
                        temperature,
                        top_p,
                        datastore,
                        num_draft,
                        token_spans,
                        filtered_ngrams,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    # if model.config.is_encoder_decoder:
                    #     output_ids = output_ids[0]
                    # else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                accept_lengths_tree_this.extend(accept_length_tree)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time, "accept_lengths:": accept_lengths_tree_this})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "category": question["category"],
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("accept_lengths_tree: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )

    # REST's hyperparameters
    parser.add_argument(
        "--datastore-path",
        type=str,
        required=True,
        help="The path of the datastore for retrival.",
    )

    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The maximum number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )
    parser.add_argument(
        "-n",
        "--ngram-n",
        type=int,
    )
    parser.add_argument(
        "-a",
        "--include-all",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--num-top-ngrams",
        type=int,
    )
    parser.add_argument(
        "-m",
        "--merge-ratio",
        type=float,
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0
        

    args.model_id = "rest-" + args.model_id+"-temperature-"+str(args.temperature)+"-top_p-"+str(args.top_p)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"llm_judge/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"llm_judge/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    ngram_datastore_settings = NGramDatastoreSettings("Aeala/ShareGPT_Vicuna_unfiltered", args.ngram_n, args.include_all, 0, args.num_top_ngrams, args.merge_ratio)

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.top_p,
        args.datastore_path,
        args.num_draft,
        args.max_token_span,
        ngram_datastore_settings
    )

    reorg_answer_file(answer_file)