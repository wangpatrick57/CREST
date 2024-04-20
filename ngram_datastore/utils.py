from typing import List, Set, Tuple
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
from collections import defaultdict


NGRAM_PICKLE_CUTOFFS = {
    1: 1597221,
    2: 1643949,
    3: 1839587,
    4: 2064567,
    5: 2171748,
}


def get_abbr_dataset_name(dataset_name: str) -> str:
        if dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            return "sharegpt"
        elif dataset_name == "bigcode/the-stack-dedup":
            return "stack"
        else:
            raise AssertionError