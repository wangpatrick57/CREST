from pathlib import Path
from draftretriever import Reader
from ngram_datastore.utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import time
import pickle

class NGramDatastore:
    def __init__(self, dataset_name: str, num_conversations: int, model_path: str, reader: Reader, ngram_n: int, num_top_ngrams: int) -> None:
        self.dataset_name = dataset_name
        self.num_conversations = num_conversations
        self.reader = reader
        self.model_path = model_path
        self.ngram_n = ngram_n
        self.num_top_ngrams = num_top_ngrams
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.datastore_dpath = Path("./ngram_datastore/built_datastores/")
        self.datastore_path = self.datastore_dpath / f"{NGramDatastore.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}-convs{num_conversations}-top{num_top_ngrams}.pkl"
        self.top0_backing_datastore_path = self.datastore_dpath / f"{NGramDatastore.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}-convs{num_conversations}-top0.pkl"
        os.makedirs(self.datastore_dpath, exist_ok=True)

    @staticmethod
    def get_abbr_dataset_name(dataset_name: str) -> str:
        if dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            return "sharegpt"
        else:
            raise AssertionError

    def build(self) -> None:
        self.datastore = dict()

        if self.dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            ngrams = get_ngrams_from_sharegpt(self.tokenizer, self.dataset_name, self.ngram_n, self.num_conversations, self.num_top_ngrams)
        elif self.dataset_name == "bigcode/the-stack":
            raise AssertionError()
        else:
            print("We only support Aeala/ShareGPT_Vicuna_unfiltered or bigcode/the-stack datasets for now")
            quit()

        if self.top0_backing_datastore_path.exists():
            print(f"Building with backing datastore {self.top0_backing_datastore_path}")
            top0_backing_datastore = NGramDatastore.load(self.top0_backing_datastore_path)
        else:
            print(f"Building with reader")
            top0_backing_datastore = None

        for ngram in tqdm(ngrams):
            # The backing datastore is equivalent to the reader and is much faster to query
            if top0_backing_datastore != None:
                tree = top0_backing_datastore[ngram]
            else:
                tree = self.reader.search(list(ngram))
            self.datastore[ngram] = tree

        with open(self.datastore_path, 'wb') as f:
            pickle.dump(self.datastore, f)
    
    def load_or_build(self) -> None:
        if os.path.exists(self.datastore_path):
            start_time = time.time()
            self.datastore = NGramDatastore.load(self.datastore_path)
            duration = time.time() - start_time
            print(f"Took {duration}s to load {self.datastore_path}")
        else:
            start_time = time.time()
            self.build()
            duration = time.time() - start_time
            print(f"Took {duration}s to build {self.datastore_path}")
    
    @staticmethod
    def load(datastore_path: str) -> dict:
        with open(datastore_path, 'rb') as f:
            return pickle.load(f)