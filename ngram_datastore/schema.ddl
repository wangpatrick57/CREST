CREATE TABLE IF NOT EXISTS ngram_datastore (
    id SERIAL PRIMARY KEY,
    ngram int32[] UNIQUE NOT NULL,
    compressed_pickled_tree bytea NOT NULL,
);