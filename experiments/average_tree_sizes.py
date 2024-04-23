import lzma
import pickle
import sys
import draftretriever
import psycopg2

if __name__ == "__main__":
    dbnames = ["sharegpt_n1_convs0_top1597221", "sharegpt_n2_convs0_top1643949", "sharegpt_n3_convs0_top1839587", "sharegpt_n4_convs0_top2064567", "sharegpt_n5_convs0_top2171748", "stack_n1_convs0_top1597221", "stack_n2_convs0_top1643949", "stack_n3_convs0_top1839587", "stack_n4_convs0_top2064567", "stack_n5_convs0_top2171748"]

    for dbname in dbnames:
        conn = psycopg2.connect(
            dbname=dbname,
            user="rest_user",
            password="rest_password",
            host="localhost",
            port=5432
        )
        cursor = conn.cursor()
        cursor.execute("SELECT compressed_pickled_tree FROM ngram_datastore ORDER BY RANDOM() LIMIT 100000;")
        compressed_pickled_tree_rows = cursor.fetchall()
        num_tokens_list = []

        for compressed_pickled_tree_row in compressed_pickled_tree_rows:
            compressed_pickled_tree = compressed_pickled_tree_row[0]
            _, _, tree_indices, _, _ = pickle.loads(lzma.decompress(compressed_pickled_tree))
            num_tokens = len(tree_indices) - 1
            num_tokens_list.append(num_tokens)
        
        average_num_tokens = sum(num_tokens_list) / len(num_tokens_list)
        print(f"{dbname} average: {average_num_tokens}")
