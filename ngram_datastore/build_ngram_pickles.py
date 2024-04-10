from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Set, Tuple
from collections import defaultdict
import pickle
from itertools import islice

NGRAM_PICKLE_CUTOFFS = {
    1: 1597221,
    2: 1643949,
    3: 1839587,
    4: 2064567,
    5: 2171748,
    6: 2290130,
}

if __name__ == "__main__":
    model_path = 'lmsys/vicuna-7b-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    one_grams = defaultdict(int)
    two_grams = defaultdict(int)
    three_grams = defaultdict(int)
    four_grams = defaultdict(int)
    five_grams = defaultdict(int)
    six_grams = defaultdict(int)

    dataset = load_dataset('Aeala/ShareGPT_Vicuna_unfiltered', split='train')

    def get_ngrams_from_list(l: List[str], n: int) -> Set[Tuple[str]]:
        return list(tuple(l[i:i+n]) for i in range(len(l) - n + 1))


    NUM_CONVERSATIONS = 0
    dataset_it = dataset if NUM_CONVERSATIONS == 0 else islice(dataset, NUM_CONVERSATIONS)

    for conversation in tqdm(dataset_it):
        for sample in conversation['conversations']:
            token_list = tokenizer.encode(sample['value'])

            one_grams_l = get_ngrams_from_list(token_list, 1)
            for one_gram in one_grams_l:
                one_grams[one_gram] += 1
            
            two_grams_l = get_ngrams_from_list(token_list, 2)
            for two_gram in two_grams_l:
                two_grams[two_gram] += 1
            
            three_grams_l = get_ngrams_from_list(token_list, 3)
            for three_gram in three_grams_l:
                three_grams[three_gram] += 1
            
            four_grams_l = get_ngrams_from_list(token_list, 4)
            for four_gram in four_grams_l:
                four_grams[four_gram] += 1

            five_grams_l = get_ngrams_from_list(token_list, 5)
            for five_gram in five_grams_l:
                five_grams[five_gram] += 1

            six_grams_l = get_ngrams_from_list(token_list, 6)
            for six_gram in six_grams_l:
                six_grams[six_gram] += 1

    one_grams = sorted(one_grams.items(), key=lambda x:-x[1])
    two_grams = sorted(two_grams.items(), key=lambda x:-x[1])
    three_grams = sorted(three_grams.items(), key=lambda x:-x[1])
    four_grams = sorted(four_grams.items(), key=lambda x:-x[1])
    five_grams = sorted(five_grams.items(), key=lambda x:-x[1])
    six_grams = sorted(six_grams.items(), key=lambda x:-x[1])

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-1gram-set-top{NGRAM_PICKLE_CUTOFFS[1]}.pkl", 'wb') as f:
        pickle.dump(one_grams[:NGRAM_PICKLE_CUTOFFS[1]], f)
    print("stored 1-gram")

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-2gram-set-top{NGRAM_PICKLE_CUTOFFS[2]}.pkl", 'wb') as f:
        pickle.dump(two_grams[:NGRAM_PICKLE_CUTOFFS[2]], f)
    print("stored 2-gram")

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-3gram-set-top{NGRAM_PICKLE_CUTOFFS[3]}.pkl", 'wb') as f:
        pickle.dump(three_grams[:NGRAM_PICKLE_CUTOFFS[3]], f)
    print("stored 3-gram")

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-4gram-set-top{NGRAM_PICKLE_CUTOFFS[4]}.pkl", 'wb') as f:
        pickle.dump(four_grams[:NGRAM_PICKLE_CUTOFFS[4]], f)
    print("stored 4-gram")

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-5gram-set-top{NGRAM_PICKLE_CUTOFFS[5]}.pkl", 'wb') as f:
        pickle.dump(five_grams[:NGRAM_PICKLE_CUTOFFS[5]], f)
    print("stored 5-gram")

    with open(f"./ngram_datastore/ngram_pickles/sharegpt-6gram-set-top{NGRAM_PICKLE_CUTOFFS[6]}.pkl", 'wb') as f:
        pickle.dump(six_grams[:NGRAM_PICKLE_CUTOFFS[6]], f)
    print("stored 6-gram")