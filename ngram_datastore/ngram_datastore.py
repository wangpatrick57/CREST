from typing import Union
from draftretriever import Reader
from ngram_datastore.utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import time
import pickle
import lzma
import psycopg2
        

class NGramDatastore:
    CREATE_STMT = """
    CREATE TABLE IF NOT EXISTS ngram_datastore (
        id SERIAL PRIMARY KEY,
        ngram integer[] UNIQUE NOT NULL,
        compressed_pickled_tree bytea NOT NULL
    )"""

    INSERT_STMT = """
    INSERT INTO ngram_datastore (ngram, compressed_pickled_tree) 
    VALUES (%s, %s)
    """

    SELECT_STMT = """
    SELECT compressed_pickled_tree FROM ngram_datastore 
    WHERE ngram = %s
    """

    def __init__(self, dbname: str):
        self.data = dict()
        self.conn = psycopg2.connect(
            dbname="postgres",
            user="rest_user",
            password="rest_password",
            host="localhost",
            port=5433
        )
        self.dbname = dbname.replace(".", "point").replace("-", "_")

    def load(self):
        self.conn.close()
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user="rest_user",
            password="rest_password",
            host="localhost",
            port=5433
        )

    def build_init(self):
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE DATABASE {self.dbname}")
        self.conn.commit()
        self.conn.close()
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user="rest_user",
            password="rest_password",
            host="localhost",
            port=5433
        )
        cursor = self.conn.cursor()
        cursor.execute(NGramDatastore.CREATE_STMT)
        cursor.close()

    def search(self, ngram):
        '''Can return either None or a tree'''
        for i in range(len(ngram)):
            tree = self.get(ngram[i:])
            if tree is not None:
                return tree
        return None
    
    def get(self, ngram):
        '''Can return either None or a tree'''
        cursor = self.conn.cursor()
        cursor.execute(NGramDatastore.SELECT_STMT, (list(ngram),))
        row = cursor.fetchone()
        cursor.close()
        compressed_pickled_tree = row[0]
        tree = pickle.loads(lzma.decompress(compressed_pickled_tree))
        return tree
    
    def insert(self, ngram, tree):
        compressed_pickled_tree = lzma.compress(pickle.dumps(tree))
        cursor = self.conn.cursor()
        cursor.execute(NGramDatastore.INSERT_STMT, (list(ngram), compressed_pickled_tree))
        cursor.close()
        self.conn.commit()

    def exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.dbname,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists


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
        self.datastore_dbname = f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}{include_all_tag}-convs{num_conversations}{discard_tag}"
        self.top0_backing_datastores = {} # a dict of backing dbnames for include-all option
        if include_all:
            for ngram in range(1, ngram_n+1):
                dbname = f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{ngram}-convs{num_conversations}-top0"
                self.top0_backing_datastores[ngram] = self.get_backing_datastore(dbname)
        else:
            dbname = f"{NGramDatastoreBuilder.get_abbr_dataset_name(dataset_name)}-n{self.ngram_n}-convs{num_conversations}-top0"
            self.top0_backing_datastores[self.ngram_n] = self.get_backing_datastore(dbname)

    @staticmethod
    def get_abbr_dataset_name(dataset_name: str) -> str:
        if dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            return "sharegpt"
        else:
            raise AssertionError

    def get_ngrams_from_dataset(self, num_ngram: int):
        print("Getting ngrams from dataset")
        if self.dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            ngrams = get_ngrams_from_sharegpt(self.tokenizer, self.dataset_name, num_ngram, self.num_conversations, self.num_top_ngrams, self.merge_ratio)
        elif self.dataset_name == "bigcode/the-stack":
            raise AssertionError()
        else:
            print("We only support Aeala/ShareGPT_Vicuna_unfiltered or bigcode/the-stack datasets for now")
            quit()
        return ngrams
    

    def get_backing_datastore(self, dbname: str):
        backing_datastore = NGramDatastore(dbname)
        if backing_datastore.exists():
            print(f"Building with backing datastore {dbname}")
            backing_datastore.load()
            return backing_datastore
        else:
            print(f"Building with reader")
            return None


    def build(self, datastore: NGramDatastore):
        datastore.build_init()

        if self.include_all:
            for num_ngram in range(1, self.ngram_n+1):
                ngrams = self.get_ngrams_from_dataset(num_ngram)
                top0_backing_datastore = self.top0_backing_datastores[num_ngram]
                for ngram in tqdm(ngrams):
                    # The backing datastore is equivalent to the reader and is much faster to query
                    if top0_backing_datastore != None:
                        tree = top0_backing_datastore.get(ngram)
                    else:
                        tree = self.reader.search(list(ngram))
                    datastore.insert(ngram, tree)
        else:
            ngrams = self.get_ngrams_from_dataset(self.ngram_n)
            top0_backing_datastore = self.top0_backing_datastores[self.ngram_n]
            for ngram in tqdm(ngrams):
                # The backing datastore is equivalent to the reader and is much faster to query
                if top0_backing_datastore != None:
                    tree = top0_backing_datastore.get(ngram)
                else:
                    tree = self.reader.search(list(ngram))
                datastore.insert(ngram, tree)
    

    def load_or_build(self) -> NGramDatastore:
        datastore = NGramDatastore(self.datastore_dbname)
        
        if datastore.exists():
            start_time = time.time()
            datastore.load()
            duration = time.time() - start_time
            print(f"Took {duration}s to load {self.datastore_dbname}")
        else:
            start_time = time.time()
            self.build(datastore)
            duration = time.time() - start_time
            print(f"Took {duration}s to build {self.datastore_dbname}")
        
        return datastore
    