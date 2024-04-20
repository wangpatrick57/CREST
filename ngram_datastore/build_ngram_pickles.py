from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Set, Tuple
from collections import defaultdict
import pickle
from itertools import islice

from datastore.get_datastore_code import get_stack_data_files
from ngram_datastore.utils import NGRAM_PICKLE_CUTOFFS, get_abbr_dataset_name


def get_ngrams_from_list(l: List[str], n: int) -> Set[Tuple[str]]:
    return list(tuple(l[i:i+n]) for i in range(len(l) - n + 1))


def store_ngram_pickles(model_path: str, dataset_name: str, max_ngram_n: int) -> None:
    ngrams = dict()
    for ngram_n in range(1, max_ngram_n + 1):
        ngrams[ngram_n] = defaultdict(int)

    data_files = get_stack_data_files(False)
    dataset = load_dataset(dataset_name, split='train', data_dir='data/python', data_files=data_files)
    NUM_CONVERSATIONS = 0
    dataset_it = dataset if NUM_CONVERSATIONS == 0 else islice(dataset, NUM_CONVERSATIONS)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for sample in tqdm(dataset_it, desc="build_ngram_pickles.store_ngram_pickles.0"):
        token_list = tokenizer.encode(sample['content'])

        for ngram_n in range(1, max_ngram_n + 1):
            this_ngrams = get_ngrams_from_list(token_list, ngram_n)
            
            for this_ngram in this_ngrams:
                ngrams[ngram_n][this_ngram] += 1
            
    for ngram_n in range(1, max_ngram_n + 1):
        ngrams[ngram_n] = sorted(ngrams[ngram_n].items(), key=lambda x:-x[1])

        with open(f"./ngram_datastore/ngram_pickles/{get_abbr_dataset_name(dataset_name)}-{ngram_n}gram-set-top{NGRAM_PICKLE_CUTOFFS[ngram_n]}.pkl", 'wb') as f:
            pickle.dump(ngrams[ngram_n][:NGRAM_PICKLE_CUTOFFS[ngram_n]], f)
        
        print(f"stored {ngram_n}-gram")


def get_ngrams_from_pickle(dataset_name, ngram_n):
    fpath = f"./ngram_datastore/ngram_pickles/{get_abbr_dataset_name(dataset_name)}-{ngram_n}gram-set-top{NGRAM_PICKLE_CUTOFFS[ngram_n]}.pkl"
    with open(fpath, "rb") as file:
        sorted_ngrams_and_counts = pickle.load(file)
        return sorted_ngrams_and_counts


if __name__ == "__main__":
    # model_path = "lmsys/vicuna-7b-v1.5"
    # dataset_name = "Aeala/ShareGPT_Vicuna_unfiltered"

    model_path = "codellama/CodeLlama-7b-instruct-hf"
    dataset_name = "bigcode/the-stack-dedup"

    store_ngram_pickles(model_path, dataset_name, 5)