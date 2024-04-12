from typing import List, Set, Tuple
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
from collections import defaultdict


# returns ngrams with specifically ngram_n number of ngrams
def get_ngrams_from_sharegpt(tokenizer: PreTrainedTokenizer, dataset_name: str, ngram_n: int, num_conversations: int, num_top_ngrams: int, merge_ratio: float) -> Set[Tuple[str]]:
    assert dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered"
    all_ngram_counts = defaultdict(int)
    dataset = load_dataset(dataset_name, split='train')

    dataset_it = dataset if num_conversations == 0 else islice(dataset, num_conversations)

    for conversations in tqdm(dataset_it):
        for sample in conversations['conversations']:
            tokens = tokenizer.encode(sample['value'])
            sample_ngrams = get_ngrams_from_list(tokens, ngram_n)
            
            for sample_ngram in sample_ngrams:
                all_ngram_counts[sample_ngram] += 1

    sorted_ngram_counts = sorted(all_ngram_counts.items(), key=(lambda item : (-item[1], item[0])))
    sorted_ngrams = [ngram for ngram, _ in sorted_ngram_counts]

    if merge_ratio != 0.0:
        top_ngrams = sorted_ngrams[:int(len(sorted_ngrams) * merge_ratio)]
    elif num_top_ngrams != 0:
        top_ngrams = sorted_ngrams[:num_top_ngrams]
    else:
        top_ngrams = sorted_ngrams
    return top_ngrams

def get_ngrams_from_list(l: List[str], ngram_n: int) -> Set[Tuple[str]]:
    return set(tuple(l[i:i+ngram_n]) for i in range(len(l) - ngram_n + 1))

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_ngram_candidates_and_draft_buffer(logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
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
    # tokens = input_ids_extend.squeeze(0)[:].to("cpu").tolist()
    # print("The token are", tokens)
    # ngrams = get_ngrams_from_list(tokens, 3)
    # for ngram in ngrams:
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
        # Retrieve draft tokens from the datastore, and get draft buffer
        # retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
        # print("the tokens are", tokens)
        # print("The tokens:", this_token)
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(tuple(this_token))
        # retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(tuple([29901, 13]))
        # print("The retrieved token list is", len(retrieved_token_list))
    
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