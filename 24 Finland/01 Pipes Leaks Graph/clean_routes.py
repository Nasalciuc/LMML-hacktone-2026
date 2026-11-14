import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE.parent / "task_finland" / "routes.csv"

# Swedish (or older) name -> Finnish name, after normalising
SV_FI = {
    "abo": "turku", "alavo": "alavus", "birkala": "pirkkala", "bjorneborg": "pori", "borga": "porvoo",
    "brahestad": "raahe", "esbo": "espoo", "fredrikshamn": "hamina", "grankulla": "kauniainen",
    "hango": "hanko", "helsingfors": "helsinki", "hogfors": "karkkila", "hyvinge": "hyvinkaa",
    "idensalmi": "iisalmi", "ikalis": "ikaalinen", "ilmola": "ilmajoki", "inga": "inkoo",
    "jakobstad": "pietarsaari", "kajana": "kajaani", "karleby": "kokkola", "kasko": "kaskinen",
    "kervo": "kerava", "keuru": "keuruu", "kimitoon": "kemionsaari", "kristinestad": "kristiinankaupunki",
    "kronoby": "kruunupyy", "kumo": "kokemaki", "kyrkslatt": "kirkkonummi", "lahtis": "lahti",
    "lappo": "lapua", "larsmo": "luoto", "lojo": "lohja", "lovisa": "loviisa", "lundo": "lieto",
    "malax": "maalahti", "korsholm": "mustasaari", "narpes": "narpio", "nadendal": "naantali",
    "nykarleby": "uusikaarlepyy", "nystad": "uusikaupunki", "pargas": "parainen", "raseborg": "raasepori",
    "raumo": "rauma", "reso": "raisio", "s:t karins": "kaarina", "s:t michel": "mikkeli", "sibbo": "sipoo",
    "sjundea": "siuntio", "tammerfors": "tampere", "tavastehus": "hameenlinna", "tornea": "tornio",
    "traskanda": "jarvenpaa", "tusby": "tuusula", "uleaborg": "oulu", "ulvsby": "ulvila", "vasa": "vaasa",
    "vanda": "vantaa", "vichtis": "vihti", "villmanstrand": "lappeenranta", "virmo": "mynamaki",
    "vora": "voyri",
}


def norm(s):
    s = s.strip().strip(".").strip().casefold()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return SV_FI.get(s, s)


def load():
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False, encoding="utf-8")
    df["a"], df["b"] = df.from_town.map(norm), df.to_town.map(norm)
    df["edge"] = [tuple(sorted(p)) for p in zip(df.a, df.b)]
    return df


def ranking(df):
    edges = sorted(set(e for e in df.edge if e[0] != e[1]))
    deg = pd.Series([t for e in edges for t in e]).value_counts()
    return edges, deg


if __name__ == "__main__":
    df = load()
    edges, deg = ranking(df)
    print("towns", deg.size, "connections", len(edges))
    print(deg.head(10).to_string())
    print("flag: SIGMOID_" + "".join(t[0].upper() for t in deg.index[:7]))
