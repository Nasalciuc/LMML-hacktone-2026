import json, sys, time, urllib.parse, urllib.request, collections
sys.stdout.reconfigure(encoding="utf-8")
DS = "Max5ive/nchlt_speech_zulu"
def get(url, tries=8):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python"}), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = e; time.sleep(2 + 2 * k)
    print("FAILED", url, err); return None
rows = []
for split in ["train", "test"]:
    off = 0
    while True:
        r = get(f"https://datasets-server.huggingface.co/search?dataset={urllib.parse.quote(DS)}&config=default&split={split}&query=ikhodi&offset={off}&length=100")
        if not r or not r["rows"]: break
        for x in r["rows"]:
            row = {k: v for k, v in x["row"].items() if k != "audio"}; row["split"] = split; row["row_idx"] = x["row_idx"]; rows.append(row)
        off += 100
        if off >= r.get("num_rows_total", 0): break
json.dump(rows, open("work/nchlt_ikhodi_rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("total rows with ikhodi:", len(rows))
c = collections.Counter(r["text"] for r in rows)
for t, n in c.most_common(): print(f"  {n:3d}  {t}")
print("female speakers:")
for r in rows:
    if r["gender"] == "female":
        print(f"  spk {r['speaker_id']} utt {r['utterance_id']} {r['duration']}s  {r['text']}")
