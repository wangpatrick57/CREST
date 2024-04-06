import click
from pathlib import Path

import draftretriever
from ngram_datastore.ngram_datastore import NGramDatastoreBuilder

@click.command()
@click.option("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
@click.option("--dataset-name", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered")
@click.option("--datastore-path", type=str, default="./datastore/datastore_chat_small.idx")
@click.option("--ngram-n", "-n", type=int, default=3)
@click.option("--num-conversations", "-c", type=int, default=10)
@click.option("--num-top-ngrams", "-t", type=int, default=10)
def main(model_path: str, dataset_name: str, datastore_path: str, ngram_n: int, num_conversations: int, num_top_ngrams: int):
    reader = draftretriever.Reader(
        index_file_path=datastore_path,
    )
    datastore_builder = NGramDatastoreBuilder(dataset_name, num_conversations, model_path, reader, ngram_n, num_top_ngrams)
    datastore = datastore_builder.load_or_build()

if __name__ == '__main__':
    main()