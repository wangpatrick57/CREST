class NGramDatastoreSettings:
    def __init__(self, dataset_name: str, ngram_n: int, include_all: bool, num_conversations: int, num_top_ngrams: int, merge_ratio: float):
        self.dataset_name = dataset_name
        self.ngram_n = ngram_n
        self.include_all = include_all
        self.num_conversations = num_conversations
        self.num_top_ngrams = num_top_ngrams
        self.merge_ratio = merge_ratio