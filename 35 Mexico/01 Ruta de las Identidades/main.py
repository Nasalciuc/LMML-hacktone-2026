import csv
import itertools
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict


def plain(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def name_tokens(r):
    return [t for t in re.findall(r"[a-z]+", plain(r["first_name"] + " " + r["last_name"]))]


NICK = {"pancho": "francisco", "paco": "francisco", "conchita": "concepcion", "pepe": "jose",
        "lupita": "guadalupe", "lupe": "guadalupe", "chuy": "jesus", "nacho": "ignacio", "memo": "guillermo",
        "lalo": "eduardo", "beto": "alberto", "toño": "antonio", "tono": "antonio", "rosy": "rosa",
        "chela": "graciela", "chava": "salvador", "quique": "enrique", "kike": "enrique", "fer": "fernando"}


def same_person(r1, r2):
    # names come abbreviated (H.), swapped (last as first), with typos or nicknames,
    # so only ask that they share a starting letter after mapping nicknames
    i1 = {NICK.get(t, t)[0] for t in name_tokens(r1)}
    i2 = {NICK.get(t, t)[0] for t in name_tokens(r2)}
    return bool(i1 & i2) or not i1 or not i2


def address_key(addr):
    return tuple(sorted(re.findall(r"\d+", addr)))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else ("test-input.csv" if os.path.exists("test-input.csv") else "train-input.csv")
    dst = sys.argv[2] if len(sys.argv) > 2 else ("test-output.json" if src.endswith("test-input.csv") else "my-train-output.json")

    rows = list(csv.DictReader(open(src, encoding="utf-8", newline="")))
    groups = defaultdict(list)
    for r in rows:
        groups[address_key(r["address"])].append(r)

    matches = []
    for g in groups.values():
        tables = [r["source_table"] for r in g]
        clash = len(tables) != len(set(tables))   # two records of one table at the same address
        for r1, r2 in itertools.combinations(g, 2):
            if r1["source_table"] == r2["source_table"]:
                continue
            if not clash or same_person(r1, r2):
                matches.append(sorted([r1["record_id"], r2["record_id"]]))
    json.dump({"matches": sorted(matches)}, open(dst, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
