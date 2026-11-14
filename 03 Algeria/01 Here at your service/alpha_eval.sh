#!/bin/bash
# usage: alpha_eval.sh RUN ALPHA  -> builds runs/RUN_aALPHA/adapter.gguf and evaluates it on public, dev and stress
R=$1; A=$2; O=runs/${R}_a$A
mkdir -p $O/peft && cp runs/$R/peft/adapter_model.safetensors $O/peft/
./.venv/Scripts/python -c "import json;c=json.load(open('runs/$R/peft/adapter_config.json'));c['lora_alpha']=$A;json.dump(c,open('$O/peft/adapter_config.json','w'))"
./.venv/Scripts/python ../task_algeria/tools/lora_to_gguf.py $O/peft --outfile $O/adapter.gguf > /dev/null
for d in public dev stress; do
  D=$([ $d = public ] && echo ../task_algeria/public || echo $d)
  ./.venv/Scripts/python ../task_algeria/evaluation_script.py --model $O/adapter.gguf --data $D --out $O/eval_$d.json > /dev/null 2> $O/eval_$d.stderr &
done
wait
