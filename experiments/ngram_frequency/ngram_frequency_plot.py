import matplotlib.pyplot as plt
import pickle
from itertools import islice
import numpy as np

# with open("./frequency_data/sharegpt-4gram-frequency-top1000.pkl", 'wb') as f:
#     pickle.dump(dict(islice(four_grams.items(), 1000)), f)
# quit()
plt.figure(figsize=(8, 4))


with open("./frequency_data/sharegpt-1gram-frequency.pkl", 'rb') as f:
    grams = pickle.load(f)

plt.subplot(2, 2, 1)
val = np.array(list(grams.values()))
x_axis = [i for i in range(len(val))]
total = np.sum(val)
cum = np.cumsum(val / total)
plt.plot(x_axis, cum)
plt.xlabel("number of 1grams")
plt.title("1-grams")
plt.xticks(fontsize="x-small")
plt.ylabel("Frequency ratio")
print(f"load 1-gram, {len(val)}")

with open("./frequency_data/sharegpt-2gram-frequency.pkl", 'rb') as f:
    grams = pickle.load(f)
plt.subplot(2, 2, 2)
val = np.array(list(grams.values()))
x_axis = [i for i in range(len(val))]
total = np.sum(val)
cum = np.cumsum(val / total)
plt.plot(x_axis, cum)
plt.xlabel("number of 2grams")
plt.title("2-grams")
plt.xticks(fontsize="x-small")
print(f"load 2-gram, {len(val)}")

with open("./frequency_data/sharegpt-3gram-frequency.pkl", 'rb') as f:
    grams = pickle.load(f)
plt.subplot(2, 2, 3)
val = np.array(list(grams.values()))
x_axis = [i for i in range(len(val))]
total = np.sum(val)
cum = np.cumsum(val / total)
plt.plot(x_axis, cum)
plt.xlabel("number of 3grams")
plt.title("3-grams")
plt.xticks(fontsize="x-small")
print(f"load 3-gram, {len(val)}")

with open("./frequency_data/sharegpt-4gram-frequency.pkl", 'rb') as f:
    grams = pickle.load(f)
plt.subplot(2, 2, 4)
val = np.array(list(grams.values()))
x_axis = [i for i in range(len(val))]
total = np.sum(val)
cum = np.cumsum(val / total)
plt.plot(x_axis, cum)
plt.xlabel("number of 4grams")
plt.title("4-grams")
plt.xticks(fontsize="x-small")
print(f"load 4-gram, {len(val)}")

# with open("./frequency_data/sharegpt-5gram-frequency.pkl", 'rb') as f:
#     grams = pickle.load(f)
# plt.subplot(2, 3, 5)
# val = np.array(list(grams.values()))
# x_axis = [i for i in range(len(val))]
# total = np.sum(val)
# cum = np.cumsum(val / total)
# plt.plot(x_axis, cum)
# plt.xlabel("number of 5grams")
# plt.title("5-grams")
# plt.xticks(fontsize="x-small")
# print(f"load 5-gram, {len(val)}")

# with open("./frequency_data/sharegpt-6gram-frequency.pkl", 'rb') as f:
#     grams = pickle.load(f)
# plt.subplot(2, 3, 6)
# val = np.array(list(grams.values()))
# x_axis = [i for i in range(len(val))]
# total = np.sum(val)
# cum = np.cumsum(val / total)
# plt.plot(x_axis, cum)
# plt.xlabel("number of 6grams")
# plt.title("6-grams")
# plt.xticks(fontsize="x-small")
# print(f"load 6-gram, {len(val)}")


plt.tight_layout()
plt.savefig("./ngram_frequency_ratio.png")
