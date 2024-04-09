from ngram_datastore.ngram_datastore_settings import NGramDatastoreSettings
import unittest


FULL_DATASTORE_SIZES = {
    "Aeala/ShareGPT_Vicuna_unfiltered": {
        1: 26346275,
        2: 1667277603,
        3: 7013245731,
        4: 9751642915,
    },
    "unittest": {
        1: 1000,
        2: 2800,
        3: 12000,
    }
}

NUM_NGRAMS = {
    "Aeala/ShareGPT_Vicuna_unfiltered": {
        1: 28598,
        2: 3720951,
        3: 16178049,
        4: 22687125,
    },
    "unittest": {
        1: 15,
        2: 40,
        3: 105,
    }
}


def estimate_datastore_size(settings: NGramDatastoreSettings) -> float:
    '''
    Returns the estimated size in bytes (the bytes can be fractional).
    '''
    assert settings.num_conversations == 0, "num_conversations should not be used in production because it basically always a bad idea for the acceptance rate / storage size tradeoff"
    dataset_datastore_sizes = FULL_DATASTORE_SIZES[settings.dataset_name]
    dataset_num_ngrams = NUM_NGRAMS[settings.dataset_name]
    total_estimated_size = 0
    ngram_ns_to_include = list(range(1, settings.ngram_n + 1)) if settings.include_all else [settings.ngram_n]

    for ngram_n in ngram_ns_to_include:
        if settings.merge_ratio != 0:
            total_estimated_size += dataset_datastore_sizes[ngram_n] * settings.merge_ratio
        elif settings.num_top_ngrams != 0:
            total_estimated_size += dataset_datastore_sizes[ngram_n] * min(1, settings.num_top_ngrams / dataset_num_ngrams[ngram_n])
        else:
            total_estimated_size += dataset_datastore_sizes[ngram_n]

    return total_estimated_size


class EstimateDatastoreSizeTests(unittest.TestCase):
    def test_single_n_no_filtering(self):
        settings = NGramDatastoreSettings("unittest", 3, False, 0, 0, 0)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 12000
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_multiple_n_no_filtering(self):
        settings = NGramDatastoreSettings("unittest", 2, True, 0, 0, 0)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 1000 + 2800
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_single_n_with_merge_ratio(self):
        settings = NGramDatastoreSettings("unittest", 1, False, 0, 0, 0.2)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 1000 * 0.2
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_multiple_n_with_merge_ratio(self):
        settings = NGramDatastoreSettings("unittest", 3, True, 0, 0, 0.3)
        estimated_size = estimate_datastore_size(settings)
        expected_size = (1000 + 2800 + 12000) * 0.3
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_single_n_with_num_top_ngrams(self):
        settings = NGramDatastoreSettings("unittest", 2, False, 0, 5, 0)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 2800 * (5 / 40)
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_multiple_n_with_num_top_ngrams(self):
        settings = NGramDatastoreSettings("unittest", 3, True, 0, 10, 0)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 1000 * (10 / 15) + 2800 * (10 / 40) + 12000 * (10 / 105)
        self.assertAlmostEqual(estimated_size, expected_size)

    def test_multiple_n_with_num_top_ngrams_some_going_over(self):
        settings = NGramDatastoreSettings("unittest", 3, True, 0, 20, 0)
        estimated_size = estimate_datastore_size(settings)
        expected_size = 1000 * (15 / 15) + 2800 * (20 / 40) + 12000 * (20 / 105)
        self.assertAlmostEqual(estimated_size, expected_size)


if __name__ == "__main__":
    unittest.main()