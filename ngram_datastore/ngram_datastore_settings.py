class NGramDatastoreSettings:
    def __init__(self, dataset_name: str, ngram_n: int, include_all: bool, num_conversations: int, num_top_ngrams: int, merge_ratio: float):
        assert not (merge_ratio != 0 and num_top_ngrams != 0), "Either set merge_ratio or num_top_ngrams but not both"
        assert 0 <= merge_ratio < 1, "merge_ratio should be [0, 1). Don't set it to 1 if you want everything, just set it to 0"
        self.dataset_name = dataset_name
        self.ngram_n = ngram_n
        self.include_all = include_all
        self.num_conversations = num_conversations
        self.num_top_ngrams = num_top_ngrams
        self.merge_ratio = merge_ratio