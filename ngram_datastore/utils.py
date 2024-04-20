import pickle
from ngram_datastore.ngram_datastore_settings import NGramDatastoreSettings


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
    
def get_filtered_ngrams(settings: NGramDatastoreSettings):
    filtered_ngrams = set()
    ngram_ns_to_include = list(range(1, settings.ngram_n + 1)) if settings.include_all else [settings.ngram_n]

    for ngram_n in ngram_ns_to_include:
        sorted_ngrams_and_counts = get_ngrams_from_pickle(settings.dataset_name, ngram_n)

        if settings.merge_ratio != 0.0:
            top_ngrams_and_counts = sorted_ngrams_and_counts[:int(len(sorted_ngrams_and_counts) * settings.merge_ratio)]
        elif settings.num_top_ngrams != 0:
            top_ngrams_and_counts = sorted_ngrams_and_counts[:settings.num_top_ngrams]
        else:
            top_ngrams_and_counts = sorted_ngrams_and_counts
        
        for top_ngram, _ in top_ngrams_and_counts:
            filtered_ngrams.add(top_ngram)
    
    return filtered_ngrams


def get_ngrams_from_pickle(dataset_name, ngram_n):
    fpath = f"./ngram_datastore/ngram_pickles/{get_abbr_dataset_name(dataset_name)}-{ngram_n}gram-set-top{NGRAM_PICKLE_CUTOFFS[ngram_n]}.pkl"
    with open(fpath, "rb") as file:
        sorted_ngrams_and_counts = pickle.load(file)
        return sorted_ngrams_and_counts