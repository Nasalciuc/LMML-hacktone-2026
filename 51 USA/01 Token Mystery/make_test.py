# builds a test-input.json with another tokenizer, to check that main.py finds it
import json
import sys

name = sys.argv[1]
calib = ("# NYC & Boston data lab — calibration\nconst greet = (who) => `Hola ${who}, ¿qué tal?`;\n"
         "price = 12.50  # USD, naïve café\n")
message = "TEST_PAYLOAD: SIGMOID_EXAMPLE decoded from New York to Los Angeles."

if name.endswith("_base") or name.endswith("_edit") or name.startswith("o200k"):
    import tiktoken
    enc = tiktoken.get_encoding(name)
    a, b = enc.encode(calib), enc.encode(message)
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    a, b = tok.encode(calib), tok.encode(message)

json.dump({"calibration": {"text": calib, "token_ids": a}, "challenge": {"token_ids": b, "token_count": len(b)}},
          open("test-input.json", "w", encoding="utf-8"), ensure_ascii=False)
