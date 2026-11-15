# isiZulu word list from Zulu Wikipedia (word frequencies). Writes work/zulu_words.tsv (word, count).
import re, collections, pandas as pd
from huggingface_hub import hf_hub_download

p = hf_hub_download("wikimedia/wikipedia", "20231101.zu/train-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(p)
cnt = collections.Counter()
for t in df["text"]:
    cnt.update(w for w in re.findall(r"[a-z]+", t.lower()) if 2 <= len(w) <= 14)
words = [(w, c) for w, c in cnt.most_common() if c >= 3]
with open("work/zulu_words.tsv", "w", encoding="utf-8") as f:
    for w, c in words:
        f.write(f"{w}\t{c}\n")
print(len(df), "articles,", len(words), "words with count>=3")
for w in ["imfihlo", "isisho", "elihle", "elisha", "isifiso", "ikhodi", "ngenisa", "iphasiwedi", "isisi", "ifu"]:
    print(w, cnt.get(w, 0))
