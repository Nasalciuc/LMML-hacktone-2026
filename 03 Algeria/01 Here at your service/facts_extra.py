"""Second batch of corpus material: years, counts, numeral drills and table-word prompts."""
from __future__ import annotations

import random

# (event clause for "In which year ...?", plain answer, year)
YEARS = [
    ("did the Second World War begin", "The Second World War began in 1939.", 1939),
    ("did the First World War end", "The First World War ended in 1918.", 1918),
    ("did the Wright brothers make their first flight", "The Wright brothers first flew in 1903.", 1903),
    ("did Napoleon lose the battle of Waterloo", "Napoleon lost at Waterloo in 1815.", 1815),
    ("did the Ottoman Turks take Constantinople", "The Ottoman Turks took Constantinople in 1453.", 1453),
    ("was the Magna Carta sealed", "The Magna Carta was sealed in 1215.", 1215),
    ("did the Russian Revolution take place", "The Russian Revolution took place in 1917.", 1917),
    ("did India become independent", "India became independent in 1947.", 1947),
    ("did the Great Depression begin", "The Great Depression began in 1929.", 1929),
    ("did Yuri Gagarin fly into space", "Yuri Gagarin flew into space in 1961.", 1961),
    ("was the Eiffel Tower completed", "The Eiffel Tower was completed in 1889.", 1889),
    ("was the Suez Canal opened", "The Suez Canal opened in 1869.", 1869),
    ("did the Chernobyl disaster happen", "The Chernobyl disaster happened in 1986.", 1986),
    ("did Nelson Mandela leave prison", "Nelson Mandela left prison in 1990.", 1990),
    ("did Shakespeare die", "Shakespeare died in 1616.", 1616),
    ("was Albert Einstein born", "Albert Einstein was born in 1879.", 1879),
    ("did Martin Luther publish his theses", "Martin Luther published his theses in 1517.", 1517),
    ("did the Battle of Hastings take place", "The Battle of Hastings took place in 1066.", 1066),
    ("did Ghana become independent", "Ghana became independent in 1957.", 1957),
    ("did the first football World Cup take place", "The first World Cup took place in 1930.", 1930),
    ("were the first modern Olympic Games held", "The first modern Olympic Games were held in 1896.", 1896),
    ("did the Soviet Union launch Sputnik", "The Soviet Union launched Sputnik in 1957.", 1957),
    ("did Mozart die", "Mozart died in 1791.", 1791),
    ("did Oran host the Mediterranean Games", "Oran hosted the Mediterranean Games in 2022.", 2022),
    ("was the Berlin Wall built", "The Berlin Wall was built in 1961.", 1961),
    ("did Barack Obama become President of the United States", "Barack Obama became President in 2009.", 2009),
    ("did Alexander Fleming discover penicillin", "Fleming discovered penicillin in 1928.", 1928),
    ("did Bell patent the telephone", "Bell patented the telephone in 1876.", 1876),
    ("did Charles Darwin publish On the Origin of Species", "Darwin published his famous book in 1859.", 1859),
    ("did Vasco da Gama reach India by sea", "Vasco da Gama reached India in 1498.", 1498),
    ("did the Black Death reach Europe", "The Black Death reached Europe in 1347.", 1347),
    ("did the Ottoman Empire come to an end", "The Ottoman Empire came to an end in 1922.", 1922),
    ("did the Great Fire of London happen", "The Great Fire of London happened in 1666.", 1666),
    ("did the great Lisbon earthquake happen", "The Lisbon earthquake happened in 1755.", 1755),
    ("was the Statue of Liberty dedicated", "The Statue of Liberty was dedicated in 1886.", 1886),
    ("did New Zealand give women the vote", "New Zealand gave women the vote in 1893.", 1893),
    ("did the Tunisian revolution begin", "The Tunisian revolution began in 2010.", 2010),
    ("did Algeria reach the last sixteen of the World Cup", "Algeria reached the last sixteen in 2014.", 2014),
    ("did euro notes and coins come into use", "Euro notes and coins came into use in 2002.", 2002),
    ("did the Spanish Armada sail against England", "The Spanish Armada sailed in 1588.", 1588),
    ("did the Hijra of the Prophet Muhammad take place", "The Hijra took place in 622.", 622),
    ("did the Western Roman Empire fall", "The Western Roman Empire fell in 476.", 476),
    ("did Charlemagne become emperor", "Charlemagne became emperor in 800.", 800),
    ("did the bombing of Hiroshima happen", "The bombing of Hiroshima happened in 1945.", 1945),
    ("did the Aswan High Dam open", "The Aswan High Dam opened in 1970.", 1970),
    ("did the Apollo eleven mission land on the Moon", "Apollo eleven landed on the Moon in 1969.", 1969),
    ("did the Arab Spring begin", "The Arab Spring began in 2010.", 2010),
    ("did the Cold War end", "The Cold War ended in 1991.", 1991),
    ("did Gandhi lead the Salt March", "Gandhi led the Salt March in 1930.", 1930),
    ("was Leonardo da Vinci born", "Leonardo da Vinci was born in 1452.", 1452),
]

MORE_NUMBERS = [
    ("How many legs does a cow have?", "A cow has four legs.", 4),
    ("How many sides does a rectangle have?", "A rectangle has four sides.", 4),
    ("How many hours are there in two days?", "There are forty-eight hours in two days.", 48),
    ("How many minutes are there in half an hour?", "There are thirty minutes in half an hour.", 30),
    ("How many months have thirty-one days?", "Seven months have thirty-one days.", 7),
    ("How many wheels does a tricycle have?", "A tricycle has three wheels.", 3),
    ("How many legs does a bird have?", "A bird has two legs.", 2),
    ("How many humps does a dromedary camel have?", "A dromedary camel has one hump.", 1),
    ("How many humps does a Bactrian camel have?", "A Bactrian camel has two humps.", 2),
    ("How many players does a handball team have on the court?", "A handball team has seven players on the court.", 7),
    ("How many cards are there in a standard deck?", "A standard deck has fifty-two cards.", 52),
    ("How many minutes does a football match last?", "A football match lasts ninety minutes.", 90),
    ("How many years are there in a millennium?", "There are 1000 years in a millennium.", 1000),
    ("How many days are there in a leap year?", "A leap year has 366 days.", 366),
    ("How many sides does a decagon have?", "A decagon has ten sides.", 10),
    ("How many zeros are there in one million?", "One million has six zeros.", 6),
    ("How many chromosomes do humans have?", "Humans have forty-six chromosomes.", 46),
    ("How many lungs does a human have?", "A human has two lungs.", 2),
    ("How many vowels are there in the English alphabet?", "The English alphabet has five vowels.", 5),
    ("How many letters are there in the Arabic alphabet?", "The Arabic alphabet has twenty-eight letters.", 28),
    ("How many days does April have?", "April has thirty days.", 30),
    ("How many days does January have?", "January has thirty-one days.", 31),
    ("How many sides does a heptagon have?", "A heptagon has seven sides.", 7),
    ("How many seconds are there in an hour?", "There are 3600 seconds in an hour.", 3600),
    ("How many players are on a cricket team?", "A cricket team has eleven players.", 11),
    ("How many quarters are there in an hour?", "There are four quarters in an hour.", 4),
    ("How many arms does a starfish usually have?", "A starfish usually has five arms.", 5),
    ("How many moons does Mars have?", "Mars has two moons.", 2),
    ("How many moons does the Earth have?", "The Earth has one moon.", 1),
    ("How many faces does a cube have?", "A cube has six faces.", 6),
    ("How many corners does a cube have?", "A cube has eight corners.", 8),
    ("How many official languages does Algeria have?", "Algeria has two official languages, Arabic and Tamazight.", 2),
    ("How many countries share a border with Algeria?", "Algeria shares borders with seven countries.", 7),
    ("How many players does an ice hockey team have on the ice?", "An ice hockey team has six players on the ice.", 6),
    ("How many sides does a triangle with equal sides have?", "An equilateral triangle has three equal sides.", 3),
    ("How many minutes are there in two hours?", "There are 120 minutes in two hours.", 120),
    ("How many days are there in two weeks?", "There are fourteen days in two weeks.", 14),
    ("How many legs do three dogs have together?", "Three dogs have twelve legs together.", 12),
]

UNIT_DRILL = [  # (small unit, big unit, factor)
    ("days", "week", 7), ("hours", "day", 24), ("minutes", "hour", 60), ("months", "year", 12),
    ("seconds", "minute", 60), ("years", "decade", 10), ("years", "century", 100), ("weeks", "year", 52),
]
NUM_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

UNCOUNTABLE = set("""gold sand water rain wind iron silver milk honey salt sugar rice bread meat cheese coffee tea wool
oil money music art history math mathematics physics chemistry science poetry peace independence culture trade time
food news truth fire earth couscous football goods cloth soup wine land sport""".split())


def numeral_drills(rng: random.Random, n_next=220, n_add=260, n_units=160, n_diff=120):
    """Plain (question, answer, number) triples that make the model spell many numbers."""
    out = []
    for _ in range(n_next):
        y = rng.randint(1000, 2030)
        q = rng.choice([f"Which year comes after {y}?", f"What year comes after {y}?", f"Which year came just before {y}?"])
        if "before" in q:
            out.append((q, f"The year before {y} was {y - 1}.", y - 1))
        else:
            out.append((q, f"The year after {y} is {y + 1}.", y + 1))
    for _ in range(n_add):
        a = rng.choice([rng.randint(10, 99), rng.randint(100, 999), rng.randint(1000, 2000)])
        b = rng.randint(1, 99)
        out.append((f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}.", a + b))
    for _ in range(n_units):
        small, big, f = rng.choice(UNIT_DRILL)
        k = rng.randint(1, 10)
        if k == 1:
            q = f"How many {small} are there in a {big}?"
            a = f"There are {f} {small} in a {big}."
        else:
            kw = NUM_WORDS[k - 2] if rng.random() < 0.5 else str(k)
            q = f"How many {small} are there in {kw} {big}s?"
            a = f"There are {f * k} {small} in {kw} {big}s."
        out.append((q, a, f * k))
    for _ in range(n_diff):
        y1 = rng.randint(1500, 1990)
        y2 = y1 + rng.randint(2, 120)
        out.append((f"How many years passed between {y1} and {y2}?", f"{y2 - y1} years passed between {y1} and {y2}.", y2 - y1))
    return out


def table_word_prompts(words: list[str]) -> list[str]:
    out = []
    for w in words:
        art = "" if w in UNCOUNTABLE else ("an " if w[0] in "aeiou" else "a ")
        out.append(f"What is {art}{w}, in one or two sentences?")
        out.append(f"Tell me one interesting thing about {art}{w}.")
    return out


def _rand_num(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.3:
        return rng.randint(0, 20)
    if u < 0.6:
        return rng.randint(21, 100)
    if u < 0.75:
        return rng.randint(101, 999)
    return rng.randint(1000, 2030)


def copy_drills(rng: random.Random, n: int = 1500):
    """Questions where the number is visible in the prompt: they teach digit -> digit name."""
    out = []
    for _ in range(n):
        a = _rand_num(rng)
        k = rng.randint(0, 7)
        if k == 0:
            q = f"Is {a} an even number?"
            ans = f"Yes, {a} is an even number." if a % 2 == 0 else f"No, {a} is an odd number."
        elif k == 1:
            b = _rand_num(rng)
            if a == b:
                b += 1
            q = f"Which is bigger, {a} or {b}?"
            ans = f"{max(a, b)} is bigger than {min(a, b)}."
        elif k == 2:
            q = f"What number comes after {a}?"
            ans = f"The number after {a} is {a + 1}."
        elif k == 3:
            a = max(a, 1)
            q = f"What number comes before {a}?"
            ans = f"The number before {a} is {a - 1}."
        elif k == 4:
            a = a % 500
            q = f"What is double {a}?"
            ans = f"Double {a} is {2 * a}."
        elif k == 5:
            a = (a % 500) * 2
            q = f"What is half of {a}?"
            ans = f"Half of {a} is {a // 2}."
        elif k == 6:
            b = rng.randint(0, min(a, 50))
            q = f"What is {a} minus {b}?"
            ans = f"{a} minus {b} is {a - b}."
        else:
            b = rng.randint(1, 9)
            q = f"What is {a} plus {b}?"
            ans = f"{a} plus {b} is {a + b}."
        out.append((q, ans, None))
    return out


ANIMALS = ("cat horse chicken duck ant bee spider crab lobster octopus snake fish frog lizard kangaroo elephant "
           "ostrich penguin camel goat sheep cow pig rabbit monkey gorilla bear lion tiger zebra giraffe beetle "
           "butterfly scorpion centipede shrimp squid dolphin whale bat owl eagle").split()
PEOPLE = ("Albert Einstein,Isaac Newton,Charles Darwin,Marie Curie,Galileo Galilei,Leonardo da Vinci,Michelangelo,"
          "William Shakespeare,Mozart,Beethoven,Johann Sebastian Bach,Napoleon Bonaparte,Abraham Lincoln,"
          "George Washington,Nelson Mandela,Mahatma Gandhi,Winston Churchill,Queen Victoria,Christopher Columbus,"
          "Victor Hugo,Charles Dickens,Mark Twain,Leo Tolstoy,Pablo Picasso,Vincent van Gogh,Claude Monet,"
          "Albert Camus,Frida Kahlo,Thomas Edison,Nikola Tesla,Alexander Graham Bell,Louis Pasteur,Sigmund Freud,"
          "Karl Marx,Martin Luther King,John F Kennedy,Barack Obama,Elvis Presley,Michael Jackson,Bob Marley,"
          "Pele,Diego Maradona,Muhammad Ali,Neil Armstrong,Yuri Gagarin,Stephen Hawking,Alan Turing,"
          "Ada Lovelace,Jane Austen,Ernest Hemingway,Emir Abdelkader,Ibn Khaldun,Ibn Battuta,Zinedine Zidane,"
          "Frederic Chopin,Johannes Gutenberg,Nicolaus Copernicus,Julius Caesar,Cleopatra,Alexander the Great").split(",")
INDEPENDENT = ("India,Pakistan,Ghana,Nigeria,Kenya,Morocco,Tunisia,Libya,Egypt,Senegal,Mali,Indonesia,Vietnam,"
               "Brazil,Mexico,Argentina,Chile,Peru,Finland,Norway,Ireland,Iceland,Philippines,Jamaica,Cuba,"
               "Angola,Mozambique,Zimbabwe,Namibia,Eritrea,South Sudan,Bangladesh,Ukraine,Estonia,Croatia").split(",")
MONTHS = "January February March April May June July August September October November December".split()


def recall_number_prompts() -> list[str]:
    qs = [f"How many legs does a {a} have?" for a in ANIMALS]
    qs += [f"In which year was {p} born?" for p in PEOPLE]
    qs += [f"In which year did {p} die?" for p in PEOPLE[:45]]
    qs += [f"In which year did {c} become independent?" for c in INDEPENDENT]
    qs += [f"How many days are there in {m}?" for m in MONTHS]
    qs += [f"How many wheels does a {v} have?" for v in "car bicycle motorcycle tricycle truck unicycle bus scooter".split()]
    qs += [f"How many strings does a {i} have?" for i in "violin cello ukulele banjo mandolin harp double bass".split()]
    qs += [f"How many players are there in a {s} team?" for s in
           "basketball volleyball rugby baseball hockey handball water polo netball cricket softball".split()]
    qs += [f"How many sides does a {p} have?" for p in "square pentagon hexagon heptagon octagon nonagon decagon dodecagon".split()]
    qs += [f"How many {a} are there in {b}?" for a, b in [
        ("seconds", "an hour"), ("minutes", "a day"), ("hours", "a week"), ("days", "a year"), ("weeks", "a year"),
        ("months", "a decade"), ("grams", "a kilogram"), ("millimetres", "a centimetre"), ("centimetres", "a metre"),
        ("metres", "a kilometre"), ("years", "a century"), ("years", "a millennium"), ("days", "a fortnight"),
        ("cents", "a euro"), ("pennies", "a pound"), ("players", "a chess game"), ("pieces", "a chess set"),
        ("cards", "a deck"), ("bones", "the human hand"), ("teeth", "a child's mouth"), ("keys", "a piano"),
        ("planets", "the solar system"), ("letters", "the Greek alphabet"), ("states", "Australia"),
        ("provinces", "Canada"), ("stripes", "the flag of the United States"), ("stars", "the flag of the European Union"),
        ("players", "a tennis doubles match"), ("holes", "a golf course"), ("events", "a decathlon")]]
    return qs


LEGS = dict(cat=4, horse=4, chicken=2, duck=2, ant=6, bee=6, spider=8, crab=10, lobster=10, octopus=8, frog=4,
            lizard=4, elephant=4, ostrich=2, penguin=2, camel=4, goat=4, sheep=4, cow=4, pig=4, rabbit=4, bear=4,
            lion=4, tiger=4, zebra=4, giraffe=4, beetle=6, butterfly=6, scorpion=8, shrimp=10, bat=2, owl=2, eagle=2)


HEADS = {
    "A": "mountain river sea lake desert island country planet star metal mineral flower tree stone region ocean volcano continent".split(),
    "B": ("animal bird insect fish author writer painter singer scientist king queen president footballer poet composer "
          "explorer philosopher tribe team snake hero actor").split(),
    "C": "city town building monument bridge tower palace mosque castle port ship car machine tool instrument vehicle museum stadium village".split(),
    "D": "month day festival holiday war battle century season game sport event year".split(),
    "E": "language book novel song poem film religion science colour dance alphabet play painting anthem".split(),
    "F": "food dish drink fruit vegetable spice sweet cake cheese bread currency coin fabric soup sauce".split(),
}
PLACES = ("Algeria France Japan Egypt Italy Brazil India Mexico Morocco China Spain Kenya Russia Turkey Greece Peru "
          "Tunisia Germany England Nigeria Iran Canada Argentina Senegal").split()


def head_call_questions(rng: random.Random, n: int = 700):
    """(question, head class) pairs: 'Which <head> is most famous in <place>?' and variants."""
    tpls = ["Which {h} is most famous in {p}?", "Which {h} is the best known in {p}?",
            "Which {h} do people in {p} love most?", "Which {h} is a symbol of {p}?"]
    out = set()
    while len(out) < n:
        cls = rng.choice(list(HEADS))
        h = rng.choice(HEADS[cls])
        out.add((rng.choice(tpls).format(h=h, p=rng.choice(PLACES)), cls))
    return sorted(out)
