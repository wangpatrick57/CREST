from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Set, Tuple
from collections import defaultdict
import pickle
from itertools import islice

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

one_cutoff = 1597221
two_cutoff = 1643949
three_cutoff = 1839587
four_cutoff = 2064567
five_cutoff = 2171748
six_cutoff = 2290130

with open(f"./ngram_datastore/ngram_pickles/sharegpt-1gram-set-top{one_cutoff}.pkl", 'wb') as f:
    pickle.dump(one_grams[:one_cutoff], f)
print("stored 1-gram")

with open(f"./ngram_datastore/ngram_pickles/sharegpt-2gram-set-top{two_cutoff}.pkl", 'wb') as f:
    pickle.dump(two_grams[:two_cutoff], f)
print("stored 2-gram")

with open(f"./ngram_datastore/ngram_pickles/sharegpt-3gram-set-top{three_cutoff}.pkl", 'wb') as f:
    pickle.dump(three_grams[:three_cutoff], f)
print("stored 3-gram")

with open(f"./ngram_datastore/ngram_pickles/sharegpt-4gram-set-top{four_cutoff}.pkl", 'wb') as f:
    pickle.dump(four_grams[:four_cutoff], f)
print("stored 4-gram")

with open(f"./ngram_datastore/ngram_pickles/sharegpt-5gram-set-top{five_cutoff}.pkl", 'wb') as f:
    pickle.dump(five_grams[:five_cutoff], f)
print("stored 5-gram")

with open(f"./ngram_datastore/ngram_pickles/sharegpt-6gram-set-top{six_cutoff}.pkl", 'wb') as f:
    pickle.dump(six_grams[:six_cutoff], f)
print("stored 6-gram")