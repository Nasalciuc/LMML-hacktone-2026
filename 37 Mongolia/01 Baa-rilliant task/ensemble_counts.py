"""Average the raw survey counts of the independently trained counters (different seeds)
and round to integers -> counts.csv (image,count)."""
import glob
import pandas as pd

files = sorted(glob.glob("counts_raw_s*.csv"))
df = None
for f in files:
    d = pd.read_csv(f).rename(columns={"count_raw": f[:-4]})
    df = d if df is None else df.merge(d, on="image")
cols = [c for c in df.columns if c != "image"]
df["count"] = df[cols].mean(axis=1).round().astype(int)
df[["image", "count"]].to_csv("counts.csv", index=False)
print("models:", files, "| total sheep", df["count"].sum(),
      "| mean |difference| between models %.2f" % (df[cols[0]] - df[cols[-1]]).abs().mean())
