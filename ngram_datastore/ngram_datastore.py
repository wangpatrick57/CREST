import os
from draftretriever import Reader
from ngram_datastore.ngram_datastore_settings import NGramDatastoreSettings
from ngram_datastore.utils import NGRAM_PICKLE_CUTOFFS, get_abbr_dataset_name, get_filtered_ngrams
from transformers import AutoTokenizer
from tqdm import tqdm

import time
import pickle
import lzma
import psycopg2



class NGramDatastore:
    CREATE_TABLE_STMT = """
    CREATE TABLE IF NOT EXISTS ngram_datastore (
        ngram integer[] PRIMARY KEY,
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
            port=5432
        )
        self.dbname = dbname.replace(".", "point").replace("-", "_")

    def load(self):
        self.conn.close()
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user="rest_user",
            password="rest_password",
            host="localhost",
            port=5432
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
            port=5432
        )
        cursor = self.conn.cursor()
        cursor.execute(NGramDatastore.CREATE_TABLE_STMT)
        cursor.close()

    def build_end(self):
        self.conn.autocommit = True
        cursor = self.conn.cursor()
        # optimize space
        cursor.execute("VACUUM FULL ngram_datastore")
        cursor.execute("REINDEX TABLE ngram_datastore")

        with open(os.path.expanduser(f"~/{self.dbname}"), "w") as f:
            pass
    
    def get_size(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT pg_total_relation_size('ngram_datastore')")
        row = cursor.fetchone()
        cursor.close()
        return int(row[0])

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

        if row == None:
            return None
        
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
                 max_ngram_n: int, num_top_ngrams: int, include_all: bool, merge_ratio: float) -> None:
        self.dataset_name = dataset_name
        self.num_conversations = num_conversations
        self.reader = reader
        self.model_path = model_path
        self.max_ngram_n = max_ngram_n
        self.num_top_ngrams = num_top_ngrams
        self.merge_ratio = merge_ratio
        discard_tag = f"-merge{merge_ratio}" if merge_ratio != 0.0 else f"-top{num_top_ngrams}"
        self.include_all = include_all
        include_all_tag = "-include-all" if include_all else ""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.datastore_dbname = f"{get_abbr_dataset_name(dataset_name)}-n{self.max_ngram_n}{include_all_tag}-convs{num_conversations}{discard_tag}"
        self.top_cutoff_backing_datastores = {} # a dict of backing dbnames for include-all option
        if include_all:
            for ngram_n in range(1, self.max_ngram_n+1):
                dbname = f"{get_abbr_dataset_name(dataset_name)}-n{ngram_n}-convs{num_conversations}-top{NGRAM_PICKLE_CUTOFFS[ngram_n]}"
                self.top_cutoff_backing_datastores[ngram_n] = self.get_backing_datastore(dbname)
        else:
            dbname = f"{get_abbr_dataset_name(dataset_name)}-n{self.max_ngram_n}-convs{num_conversations}-top{NGRAM_PICKLE_CUTOFFS[self.max_ngram_n]}"
            self.top_cutoff_backing_datastores[self.max_ngram_n] = self.get_backing_datastore(dbname)
    

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
            for ngram_n in range(1, self.max_ngram_n+1):
                # We pass False into self.include_all because we only want the n-grams of this value of n
                ngram_datastore_settings = NGramDatastoreSettings(self.dataset_name, ngram_n, False, self.num_conversations, self.num_top_ngrams, self.merge_ratio)
                ngrams = get_filtered_ngrams(ngram_datastore_settings)
                top_cutoff_backing_datastore: NGramDatastore = self.top_cutoff_backing_datastores[ngram_n]
                for ngram in tqdm(ngrams, desc="ngram_datastore.NGramDatastoreBuilder.build.0"):
                    try:
                        # The backing datastore is equivalent to the reader and is much faster to query
                        if top_cutoff_backing_datastore != None:
                            tree = top_cutoff_backing_datastore.get(ngram)

                            # Just in case it's not there
                            if tree == None:
                                tree = self.reader.search(list(ngram))
                        else:
                            tree = self.reader.search(list(ngram))
                        datastore.insert(ngram, tree)
                    except ValueError:
                        # This is possible if cut_to_choices() doesn't cut it enough. See
                        # lib.rs for more details.
                        print("Encountered ValueError from search(). This is okay though.")
                    except Exception:
                        raise
        else:
            ngram_datastore_settings = NGramDatastoreSettings(self.dataset_name, self.max_ngram_n, self.include_all, self.num_conversations, self.num_top_ngrams, self.merge_ratio)
            ngrams = get_filtered_ngrams(ngram_datastore_settings)
            top_cutoff_backing_datastore = self.top_cutoff_backing_datastores[self.max_ngram_n]
            for ngram in tqdm(ngrams, desc="ngram_datastore.NGramDatastoreBuilder.build.1"):
                # The backing datastore is equivalent to the reader and is much faster to query
                if top_cutoff_backing_datastore != None:
                    tree = top_cutoff_backing_datastore.get(ngram)
                    datastore.insert(ngram, tree)
                else:
                    try:
                        tree = self.reader.search(list(ngram))
                        datastore.insert(ngram, tree)
                    except ValueError:
                        # This is possible if cut_to_choices() doesn't cut it enough. See
                        # lib.rs for more details.
                        print("Encountered ValueError from search(). This is okay though.")
                    except Exception:
                        raise
        
        datastore.build_end()
    

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
    