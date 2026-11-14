"""Egypt - Always Valid, No Excuses.

Reads test-input.txt (beside this file), asks the local Qwen3-1.7B model to turn each
OCR record into one line of JSON, checks every answer against the format rules and,
when a check fails, sends the model the exact problems so it corrects its own answer.
Writes test-output.txt (line N = record N).

Usage:
    python main.py                                  # test-input.txt -> test-output.txt
    python main.py IN OUT [EXPECTED]                # any file; EXPECTED scores the result
"""
import json
import os
import re
import sys
import time


def _add_cuda_dll_dirs():
    """Windows only: let a prebuilt CUDA wheel find CUDA runtime DLLs shipped as pip packages."""
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        root = os.path.join(sp, "nvidia")
        if os.path.isdir(root):
            for sub in os.listdir(root):
                b = os.path.join(root, sub, "bin")
                if os.path.isdir(b):
                    os.add_dll_directory(b)
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()
from llama_cpp import Llama, LlamaGrammar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen3-1.7B-Q8_0.gguf")
MAX_FIXES = 6

SYSTEM = """You convert one OCR'd museum record card into ONE line of strict JSON.

The record is a list of segments separated by "|".
1. Split on "|". Trim spaces at both ends of every segment, key and value.
2. A segment is a real field ONLY if it looks like `key: value` where the key is a single simple word.
   Everything else is scan noise and must be DROPPED: page numbers (page 22, p. 128, 22, 128),
   headers (Chapter IX), footers (- 45 -), bracketed marks ([torn], [illegible], (faded), (margin note)),
   figure labels (Fig 3, Plate II), garble (###, ~~~, * * *), remarks without a colon (end of page, scanned 1993, see above).
3. Only the FIRST ":" separates key from value. Later colons belong to the value: `ratio: 3:1` -> "ratio":"3:1",
   `epithet: cost: $200` -> "epithet":"cost: $200".
4. Value types (after trimming):
   - empty value -> null
   - exactly true / false -> JSON true / false (no quotes)
   - a whole number such as 40, 0, -3 -> JSON number (no quotes)
   - anything else -> JSON string, copied character for character; escape " as \\" and \\ as \\\\.
5. Keep every real field, in input order, with the key exactly as written. Add nothing, invent nothing.
Answer with the JSON object only."""

# Few-shot pairs taken from the sample (train-input.txt / train-output.txt).
EXAMPLES = [
    ("page 22| king :Menes| title: ruled with \"the Aten\" |[torn]| alive :true | Chapter IX",
     '{"king":"Menes","title":"ruled with \\"the Aten\\"","alive":true}'),
    (" reign: true  |  (faded)  |  ###  |  era : Seti I  |  title: temple @ Karnak  |  rank : -18   |  height:    |  god:-57  |  page 7  |  city: Amun",
     '{"reign":true,"era":"Seti I","title":"temple @ Karnak","rank":-18,"height":null,"god":-57,"city":"Amun"}'),
    ("god: Thoth | title: \"lord of the moon\" | scroll: C:\\papyri\\thoth | - 12 -",
     '{"god":"Thoth","title":"\\"lord of the moon\\"","scroll":"C:\\\\papyri\\\\thoth"}'),
    ("ratio: 2:1 | width: 230 | height:146 | material: limestone & granite | Plate VII",
     '{"ratio":"2:1","width":230,"height":146,"material":"limestone & granite"}'),
    ("king:  temple @ Karnak | cult: star * of Amun| era :Akhenaten| epithet : cost: $200 | ratio: Horemheb| Fig 3",
     '{"king":"temple @ Karnak","cult":"star * of Amun","era":"Akhenaten","epithet":"cost: $200","ratio":"Horemheb"}'),
    ("  queen:Nefertari| age : 0 |rank:  | (margin) | city :Luxor  ",
     '{"queen":"Nefertari","age":0,"rank":null,"city":"Luxor"}'),
]

# Flat JSON object: string keys; values string / integer / true / false / null.
# Only the two escapes the data needs (\" and \\) are allowed inside strings.
GRAMMAR = r'''
root   ::= "{" ( pair ( "," pair )* )? "}"
pair   ::= string ":" value
value  ::= string | number | "true" | "false" | "null"
string ::= "\"" char* "\""
char   ::= [^"\\\x00-\x1f] | "\\\"" | "\\\\"
number ::= "-"? ( "0" | [1-9] [0-9]* )
'''

# ---------------------------------------------------------------- rule checker
KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INT_RE = re.compile(r"-?\d+")


def convert(v):
    if v == "":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if INT_RE.fullmatch(v):
        return int(v)
    return v


def rule_fields(record):
    """Fields the rules say the record holds (used only to check the model's answer)."""
    out = {}
    for seg in record.split("|"):
        if ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        k = k.strip()
        if KEY_RE.fullmatch(k):
            out[k] = convert(v.strip())
    return out


def same(a, b):
    return type(a) is type(b) and a == b


def describe(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "the boolean " + json.dumps(v)
    if isinstance(v, int):
        return "the number " + json.dumps(v)
    return "the string " + json.dumps(v, ensure_ascii=False)


def problems(record, text):
    """Check the model's answer against the rules.
    Returns (errors, edits): human-readable problems and the matching edit instructions.
    Both empty = the line is valid. edits is None when the answer is not a JSON object."""
    try:
        got = json.loads(text)
    except Exception as e:  # noqa: BLE001
        return [f"the answer is not valid JSON ({e})"], None
    if not isinstance(got, dict):
        return ["the answer must be one JSON object"], None
    want = rule_fields(record)
    errs, edits = [], []
    for k, v in want.items():
        lit = json.dumps(v, ensure_ascii=False)
        if k not in got:
            errs.append(f'field "{k}" is missing; it must be {describe(v)}')
            edits.append(("add", k, lit))
        elif not same(got[k], v):
            errs.append(f'field "{k}" must be {describe(v)}, not {describe(got[k])}')
            edits.append(("change", k, lit))
    for k in got:
        if k not in want:
            errs.append(f'"{k}" is not a field of the record (scan noise or invented)')
            edits.append(("delete", k, None))
    return errs, edits


# ---------------------------------------------------------------- model
def chatml(messages):
    parts = [f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages]
    # Qwen3 non-thinking mode: empty think block, exactly as its chat template does.
    return "".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def base_messages(record):
    msgs = [("system", SYSTEM)]
    for inp, out in EXAMPLES:
        msgs.append(("user", f"Record: {inp} /no_think"))
        msgs.append(("assistant", "<think>\n\n</think>\n\n" + out))
    msgs.append(("user", f"Record: {record} /no_think"))
    return msgs


def ask(llm, messages, grammar=None, temperature=0.0, seed=0):
    r = llm.create_completion(
        chatml(messages), max_tokens=400, temperature=temperature,
        top_k=1 if temperature == 0 else 40, top_p=1.0 if temperature == 0 else 0.95,
        repeat_penalty=1.0, grammar=grammar, stop=["<|im_end|>", "\n"], seed=seed)
    return r["choices"][0]["text"].strip()


FIX_SYSTEM = "You edit one JSON object exactly as told. Output the edited JSON object only, on one line."


def edit_messages(text, edits, style=0):
    """Ask the model to apply the checker's add/change instructions to its own answer.
    Two phrasings; the loop alternates them when one does not get through."""
    cur = json.loads(text)
    lines = []
    for op, k, lit in edits:
        if op == "add":
            lines.append(f'Add the key "{k}" with value {lit}.')
        elif op == "change":
            old = json.dumps(cur[k], ensure_ascii=False)
            if style % 2 == 0:
                lines.append(f'In this JSON, replace the text "{k}":{old} with "{k}":{lit}')
            else:
                lines.append(f'The value of "{k}" is wrong. Write {lit} exactly: "{k}":{lit}')
    return [("system", FIX_SYSTEM),
            ("user", "JSON: " + text + "\n" + "\n".join(lines) + " /no_think")]


def keep_messages(text, keys):
    """Ask the model to drop the keys that are scan noise (it copes better with a keep-list)."""
    return [("system", FIX_SYSTEM),
            ("user", "JSON: " + text + "\nKeep only these keys, in this order: "
             + ", ".join(f'"{k}"' for k in keys) + ". Copy their values unchanged. /no_think")]


def convert_record(llm, grammar, record):
    """First pass: the model converts the record. Then, while the checker finds problems,
    the model is told exactly what is wrong and fixes its own answer."""
    text = ask(llm, base_messages(record))
    errs, edits = problems(record, text)
    fixes, seen = 0, {text}
    while errs and fixes < MAX_FIXES:
        fixes += 1
        if edits is None:
            # not JSON at all: convert again, this time under the JSON grammar
            # (the grammar is slow - it masks the whole vocabulary per token - so only here)
            msgs = base_messages(record)
            new = ask(llm, msgs, grammar)
        else:
            if any(op == "delete" for op, _, _ in edits):
                msgs = keep_messages(text, list(rule_fields(record)))
            else:
                msgs = edit_messages(text, edits, style=fixes - 1)
            new = ask(llm, msgs)
        if new in seen:  # stuck on an answer it already gave: let it sample a little
            new = ask(llm, msgs, grammar if edits is None else None, temperature=0.5, seed=fixes)
        seen.add(new)
        new_errs, new_edits = problems(record, new)
        if new_edits is not None or edits is None:
            text, errs, edits = new, new_errs, new_edits
    return text, fixes, errs


def main():
    args = sys.argv[1:]
    src = args[0] if len(args) > 0 else os.path.join(HERE, "test-input.txt")
    dst = args[1] if len(args) > 1 else os.path.join(HERE, "test-output.txt")
    gold = args[2] if len(args) > 2 else None

    with open(src, encoding="utf-8") as f:
        records = f.read().split("\n")
    if records and records[-1] == "":
        records.pop()

    t0 = time.time()
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, n_batch=1024,
                flash_attn=True, seed=0, verbose=False)
    grammar = LlamaGrammar.from_string(GRAMMAR, verbose=False)

    out_lines, fixed, failed = [], 0, []
    for i, rec in enumerate(records, 1):
        text, fixes, errs = convert_record(llm, grammar, rec)
        fixed += fixes > 0
        if errs:
            failed.append(i)
            print(f"[line {i}] still wrong after {fixes} fixes: {errs}", file=sys.stderr)
        elif fixes:
            print(f"[line {i}] corrected by the model after {fixes} round(s)", file=sys.stderr)
        out_lines.append(text.replace("\n", " "))

    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_lines) + "\n")

    n = len(records)
    print(f"{n} records, {n - len(failed)} pass the format check, "
          f"{fixed} needed a correction round, {time.time() - t0:.1f}s", file=sys.stderr)

    if gold:
        with open(gold, encoding="utf-8") as f:
            exp = [json.loads(x) for x in f.read().split("\n") if x.strip()]
        ok = 0
        for i, (line, e) in enumerate(zip(out_lines, exp), 1):
            try:
                g = json.loads(line)
                good = g == e and all(same(g[k], e[k]) for k in e)
            except Exception:  # noqa: BLE001
                good = False
            ok += good
            if not good:
                print(f"[line {i}] MISMATCH\n  got: {line}\n  exp: {json.dumps(e)}", file=sys.stderr)
        print(f"score vs expected: {ok}/{len(exp)}", file=sys.stderr)


if __name__ == "__main__":
    main()
