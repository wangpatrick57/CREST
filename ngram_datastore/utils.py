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