import os, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bioacoustics_model_zoo as bmz
from sklearn.decomposition import PCA

BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
os.makedirs("samples", exist_ok=True)

# 1. Download a small sample: 8 clips from each of 3 categories
meta = pd.read_csv(f"{BASE}/meta/esc50.csv")
categories = ["engine", "sea_waves", "rain"]
picked = pd.concat([meta[meta.category == c].head(8) for c in categories])

paths, labels = [], []
for _, row in picked.iterrows():
    local = f"samples/{row.filename}"
    if not os.path.exists(local):
        urllib.request.urlretrieve(f"{BASE}/audio/{row.filename}", local)
    paths.append(local)
    labels.append(row.category)

# 2. Load Perch 2.0 (first run downloads the model from Kaggle)
perch = bmz.Perch2()

# 3. Get one embedding per 5-second window
emb = perch.embed(paths)
X = np.asarray(emb)
print(X.shape)  # expect (24, 1536)
assert len(X) == len(labels), "more than one window per clip"

# 4. Plot the embeddings in 2D and save the image
coords = PCA(n_components=2).fit_transform(X)
for c in categories:
    idx = [i for i, l in enumerate(labels) if l == c]
    plt.scatter(coords[idx, 0], coords[idx, 1], label=c)
plt.legend()
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Perch 2.0 embeddings of sample clips")
plt.savefig("embeddings.png", dpi=150)
print("saved embeddings.png")
