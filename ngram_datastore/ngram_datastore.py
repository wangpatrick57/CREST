from pathlib import Path
from draftretriever import Reader
from ngram_datastore.utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import time
import pickle
        

class NGramDatastore:
    def __init__(self):
        self.data = dict()

    def search(self, ngram):
        '''Can return either None or a tree'''
        for i in range(len(ngram)):
            tree = self.get(ngram[i:])
            if tree is not None:
                return tree
        return None
    
    def get(self, ngram):
        '''Can return either None or a tree'''
        return self.data[ngram]
    
    def insert(self, ngram, tree):
        self.data[ngram] = tree


class NGramDatastoreBuilder:
    EXTENSION = 'pkl'

    def __init__(self, dataset_name: str, num_conversations: int, model_path: str, reader: Reader, 
                 ngram_n: int, num_top_ngrams: int, include_all: bool, merge_ratio: float) -> None:
        self.dataset_name = dataset_name
        self.num_conversations = num_conversations
        self.reader = reader
        self.model_path = model_path
        self.ngram_n = ngram_n
        self.num_top_ngrams = num_top_ngrams
        self.merge_ratio = merge_ratio
        discard_tag = f"-merge{merge_ratio}" if merge_ratio != 0.0 else f"-top{num_top_ngrams}"
        self.include_all = include_all
        include_all_tag = "-include-all" if include_all else ""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.datastore_dpath = Path("./ngram_datastore/built_datastores/")
        self.datastore_path = self.datastore_dpath / f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}{include_all_tag}-convs{num_conversations}{discard_tag}.{NGramDatastoreBuilder.EXTENSION}"
        self.top0_backing_datastore_path = {}   # a dict of backing paths for include-all option
        if include_all:
            for ngram in range(1, ngram_n+1):
                path = self.datastore_dpath / f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{ngram}-convs{num_conversations}-top0.{NGramDatastoreBuilder.EXTENSION}"
                self.top0_backing_datastore_path[ngram] = path
        else:
            path = self.datastore_dpath / f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}-convs{num_conversations}-top0.{NGramDatastoreBuilder.EXTENSION}"
            self.top0_backing_datastore_path[self.ngram_n] = path

        os.makedirs(self.datastore_dpath, exist_ok=True)

    @staticmethod
    def get_abbr_dataset_name(dataset_name: str) -> str:
        if dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            return "sharegpt"
        else:
            raise AssertionError

    def get_ngrams_from_dataset(self, num_ngram: int):
        if self.dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            ngrams = get_ngrams_from_sharegpt(self.tokenizer, self.dataset_name, num_ngram, self.num_conversations, self.num_top_ngrams, self.merge_ratio)
        elif self.dataset_name == "bigcode/the-stack":
            raise AssertionError()
        else:
            print("We only support Aeala/ShareGPT_Vicuna_unfiltered or bigcode/the-stack datasets for now")
            quit()
        return ngrams
    

    def get_backing_datastore(self, path: str):
        if path.exists():
            print(f"Building with backing datastore {path}")
            top0_backing_datastore = NGramDatastoreBuilder.load(path)
        else:
            print(f"Building with reader")
            top0_backing_datastore = None
        return top0_backing_datastore


    def build(self) -> NGramDatastore:
        datastore = NGramDatastore()

        if self.include_all:
            for num_ngram in range(1, self.ngram_n+1):
                ngrams = self.get_ngrams_from_dataset(num_ngram)
                top0_backing_datastore = self.get_backing_datastore(self.top0_backing_datastore_path[num_ngram])
                for ngram in tqdm(ngrams):
                    # The backing datastore is equivalent to the reader and is much faster to query
                    if top0_backing_datastore != None:
                        tree = top0_backing_datastore.get(ngram)
                    else:
                        tree = self.reader.search(list(ngram))
                    datastore.insert(ngram, tree)
        else:
            ngrams = self.get_ngrams_from_dataset(self.ngram_n)
            top0_backing_datastore = self.get_backing_datastore(self.top0_backing_datastore_path[self.ngram_n])
            for ngram in tqdm(ngrams):
                # The backing datastore is equivalent to the reader and is much faster to query
                if top0_backing_datastore != None:
                    tree = top0_backing_datastore[ngram]
                else:
                    tree = self.reader.search(list(ngram))
                datastore.insert(ngram, tree)

        with open(self.datastore_path, 'wb') as f:
            pickle.dump(datastore, f)
    

    def load_or_build(self) -> NGramDatastore:
        if os.path.exists(self.datastore_path):
            start_time = time.time()
            datastore = NGramDatastoreBuilder.load(self.datastore_path)
            duration = time.time() - start_time
            print(f"Took {duration}s to load {self.datastore_path}")
        else:
            start_time = time.time()
            datastore = self.build()
            duration = time.time() - start_time
            print(f"Took {duration}s to build {self.datastore_path}")
        
        return datastore
    
    @staticmethod
    def load(datastore_path: str) -> NGramDatastore:
        with open(datastore_path, 'rb') as f:
            return pickle.load(f)