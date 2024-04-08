import matplotlib.pyplot as plt
import pickle
from itertools import islice

# with open("./frequency_data/sharegpt-4gram-frequency-top1000.pkl", 'wb') as f:
#     pickle.dump(dict(islice(four_grams.items(), 1000)), f)
# quit()

with open("./frequency_data/sharegpt-1gram-frequency.pkl", 'rb') as f:
    one_grams = pickle.load(f)
    # print(len(one_grams.keys()))
print("load 1-gram")
with open("./frequency_data/sharegpt-2gram-frequency.pkl", 'rb') as f:
    two_grams = pickle.load(f)
print("load 2-gram")
with open("./frequency_data/sharegpt-3gram-frequency.pkl", 'rb') as f:
    three_grams = pickle.load(f)
print("load 3-gram")
with open("./frequency_data/sharegpt-4gram-frequency-top1000.pkl", 'rb') as f:
    four_grams = pickle.load(f)
print("load 4-gram")
with open("./frequency_data/sharegpt-5gram-frequency-top10000.pkl", 'rb') as f:
    five_grams = pickle.load(f)
print("load 5-gram")
with open("./frequency_data/sharegpt-6gram-frequency-top10000.pkl", 'rb') as f:
    six_grams = pickle.load(f)
print("load 6-gram")


plt.figure(figsize=(10, 4))
topk = 100
x_axis = [i for i in range(topk)]
plt.subplot(2, 3, 1)
plt.bar(x_axis, list(one_grams.values())[:topk])
plt.title("1-grams")
plt.ylabel("Frequency")

plt.subplot(2, 3, 2)
plt.bar(x_axis, list(two_grams.values())[:topk])
plt.title("2-grams")

plt.subplot(2, 3, 3)
plt.bar(x_axis, list(three_grams.values())[:topk])
plt.title("3-grams")

plt.subplot(2, 3, 4)
plt.bar(x_axis, list(four_grams.values())[:topk])
plt.title("4-grams")
plt.ylabel("Frequency")

plt.subplot(2, 3, 5)
plt.bar(x_axis, list(five_grams.values())[:topk])
plt.title("5-grams")

plt.subplot(2, 3, 6)
plt.bar(x_axis, list(six_grams.values())[:topk])
plt.title("6-grams")

plt.tight_layout()
plt.savefig("./ngram_frequency.png")