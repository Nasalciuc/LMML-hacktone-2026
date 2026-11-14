import json
import os
import sys

TIKTOKEN = ["r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base", "o200k_harmony"]

HF = [
    "gpt2", "openai-community/gpt2", "EleutherAI/gpt-neox-20b", "EleutherAI/pythia-160m",
    "bert-base-uncased", "bert-base-cased", "bert-base-multilingual-cased", "bert-base-multilingual-uncased",
    "distilbert-base-uncased", "roberta-base", "distilroberta-base", "facebook/bart-base",
    "xlm-roberta-base", "albert-base-v2", "xlnet-base-cased", "t5-small", "google/flan-t5-small",
    "google/mt5-small", "google/byt5-small", "google/electra-small-discriminator", "microsoft/deberta-v3-base",
    "allenai/longformer-base-4096", "facebook/opt-125m", "bigscience/bloom-560m", "tiiuae/falcon-7b",
    "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct", "Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B",
    "mistralai/Mistral-7B-v0.1", "hf-internal-testing/llama-tokenizer", "huggyllama/llama-7b",
    "NousResearch/Llama-2-7b-hf", "NousResearch/Meta-Llama-3-8B", "unsloth/Llama-3.2-1B",
    "codellama/CodeLlama-7b-hf", "bigcode/starcoder2-3b", "Salesforce/codegen-350M-mono",
    "deepseek-ai/deepseek-coder-1.3b-base", "stabilityai/stablelm-2-1_6b", "HuggingFaceTB/SmolLM2-135M",
    "allenai/OLMo-1B-hf", "google/gemma-2b", "unsloth/gemma-2-2b", "facebook/xglm-564M",
    "sentence-transformers/all-MiniLM-L6-v2", "microsoft/mpnet-base", "camembert-base",
]


def load_input():
    for name in ("test-input.json", "train-input.json"):
        if os.path.exists(name):
            return name, json.load(open(name, encoding="utf-8"))
    sys.exit("no input file")


def tiktoken_candidates():
    try:
        import tiktoken
    except ImportError:
        return
    names = list(TIKTOKEN)
    try:
        names += [n for n in tiktoken.list_encoding_names() if n not in names]
    except Exception:
        pass
    for name in names:
        try:
            enc = tiktoken.get_encoding(name)
        except Exception:
            continue
        yield (name,
               lambda t, e=enc: [e.encode(t, allowed_special="all")],
               lambda ids, e=enc: e.decode(ids))


def hf_candidates():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    for name in HF:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception:
            continue
        def enc(t, tok=tok):
            outs = []
            for special in (False, True):
                try:
                    outs.append(tok.encode(t, add_special_tokens=special))
                except Exception:
                    pass
            return outs
        def dec(ids, tok=tok):
            return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        yield name, enc, dec


def sp_candidates():
    try:
        import sentencepiece as spm
    except ImportError:
        return
    for f in os.listdir("."):
        if f.endswith(".model"):
            try:
                sp = spm.SentencePieceProcessor(model_file=f)
            except Exception:
                continue
            yield (os.path.splitext(f)[0],
                   lambda t, s=sp: [s.encode(t), [s.bos_id()] + s.encode(t)],
                   lambda ids, s=sp: s.decode(ids))


def main():
    name_in, data = load_input()
    text = data["calibration"]["text"]
    target = list(data["calibration"]["token_ids"])
    challenge = list(data["challenge"]["token_ids"])

    found = None
    for source in (tiktoken_candidates, hf_candidates, sp_candidates):
        for name, enc, dec in source():
            try:
                if any(list(ids) == target for ids in enc(text)):
                    found = (name, dec)
                    break
            except Exception:
                continue
        if found:
            break
    if not found:
        sys.exit("no tokenizer reproduces the calibration ids")

    name, dec = found
    out = {"tokenizer_id": name, "decoded_message": dec(challenge)}
    out_name = "test-output.json" if name_in == "test-input.json" else "my-train-output.json"
    json.dump(out, open(out_name, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
