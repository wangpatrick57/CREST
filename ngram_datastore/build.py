import click
from pathlib import Path

import draftretriever
from ngram_datastore.utils import NGRAM_PICKLE_CUTOFFS
from ngram_datastore.ngram_datastore import NGramDatastoreBuilder

@click.command()
@click.option("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
@click.option("--dataset-name", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered")
@click.option("--datastore-path", type=str, default="./datastore/datastore_chat_small.idx")
@click.option("--ngram-n", "-n", type=int, default=3)               # number of ngrams to build the datastore on
@click.option('--include-all', '-a', is_flag=True)                  # includes all ngrams up to ngram_n. Specify either num_top_ngrams or merge_ratio for keeping only a few when merging
@click.option("--num-conversations", "-c", type=int, default=0)    # number of conversations to build the datstore
@click.option("--num-top-ngrams", "-t", type=int, default=0)       # for keeping in the datastore
@click.option('--merge-ratio', '-r',type=float, default=0.0)        # merge ratio. If not specified, merging defaults to choosing top N
def main(model_path: str, dataset_name: str, datastore_path: str, ngram_n: int, num_conversations: int, num_top_ngrams: int, include_all: bool, merge_ratio: float):
    if num_top_ngrams == 0:
        num_top_ngrams = NGRAM_PICKLE_CUTOFFS[ngram_n]
    reader = draftretriever.Reader(
        index_file_path=datastore_path,
    )
    datastore_builder = NGramDatastoreBuilder(dataset_name, num_conversations, model_path, reader, ngram_n, num_top_ngrams, include_all, merge_ratio)
    datastore = datastore_builder.load_or_build()
    print(datastore.get_size())

if __name__ == '__main__':
    main()