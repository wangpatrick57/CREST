from pathlib import Path
from draftretriever import Reader
from ngram_datastore.utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import time
import pickle
        

class NGramDatastore:
    def __init__(self, unique_id: str, should_load: bool):
        self.data = dict()
        if should_load:
            with open(unique_id, 'rb') as f:
                self.data = pickle.load(f).data
        
        # print("the keys are:")
        # for key in self.data.keys():
        #     print(key)

    def search(self, ngram):
        '''Can return either None or a tree'''
        # for i in range(len(ngram)):
        #     tree = self.get(ngram[i:])
        #     if tree is not None:
        #         return tree
        # return None
        # print("looking for the ngram", ngram)
        if ngram in self.data:
            # print("Found the ngram")
            tree = self.get(ngram)
            # print("The length of the retrieved tree is", len(tree[0]))
            # print("The retrieved tree is", tree[0])
            return tree
        return [], [], [], [], []

        # print("looking for the index")
        # ngram_key = tuple(ngram)
        # if ngram in self.data:
        #     # print("found the ngram in the datastore", ngram_key)
        #     tree = self.get(ngram)
        #     print("The length of the retrieved tree is", len(tree[0]))
        #     print("The retrieved tree is", tree[0])
        #     return self.get(ngram)
        # return [[], [], [], [], []]
        # ngram_key = tuple(ngram)
        # if (ngram_key) in self.data:
        #     print("found the ngram in the datastore", ngram_key)
        #     tree = self.get(ngram)
        #     print("The length of the retrieved tree is", len(tree[0]))
        #     print("The retrieved tree is", tree[0])
        #     return self.get(ngram)
        # return [[], [], [], [], []]
    
    def get(self, ngram):
        '''Can return either None or a tree'''
        return self.data[ngram]
    
    def insert(self, ngram, tree):
        self.data[ngram] = tree

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    

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
        self.datastore_path = "/home/ubuntu/REST/ngram_datastore/built_datastores/sharegpt-n1-convs0-top0.pkl"
        # self.datastore_path = self.datastore_dpath / f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}{include_all_tag}-convs{num_conversations}{discard_tag}.{NGramDatastoreBuilder.EXTENSION}"
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
            top0_backing_datastore = NGramDatastore(path, True)
        else:
            print(f"Building with reader")
            top0_backing_datastore = None
        return top0_backing_datastore


    def build(self) -> NGramDatastore:
        datastore = NGramDatastore(None, False)
        print("CAlled the build function")

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
            print("include all false")
            ngrams = self.get_ngrams_from_dataset(self.ngram_n)

            print("finished getting the ngrams from dataset")
            top0_backing_datastore = self.get_backing_datastore(self.top0_backing_datastore_path[self.ngram_n])
            # first_ngram = ngrams[0]
            # print("The index of the first ngram value", first_ngram)
            # first_ngram_tree = self.reader.search(list(first_ngram))
            # print(type("the first ngram tree", first_ngram_tree))
            # first_ngram_tree = []
            # count = 0
            # for ngram in tqdm(ngrams):
            #     if count == 0:
            #         count = 1
            #         first_ngram_tree = self.reader.search(list(ngram))
            #     else:
            #         datastore.insert(ngram, first_ngram_tree)

            for ngram in tqdm(ngrams):
                # The backing datastore is equivalent to the reader and is much faster to query
                if top0_backing_datastore != None:
                    tree = top0_backing_datastore.get(ngram)
                else:
                    tree = self.reader.search(list(ngram))
                datastore.insert(ngram, tree)

        datastore.save(self.datastore_path)
        return datastore
    

    def load_or_build(self) -> NGramDatastore:
        if os.path.exists(self.datastore_path):
            print("Loading the datastore from exisitng datastore file")
            start_time = time.time()
            datastore = NGramDatastore(self.datastore_path, True)
            duration = time.time() - start_time
            print(f"Took {duration}s to load {self.datastore_path}")
        else:
            start_time = time.time()
            datastore = self.build()
            duration = time.time() - start_time
            print(f"Took {duration}s to build {self.datastore_path}")
        
        return datastore
    