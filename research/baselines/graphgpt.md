# GraphGPT

- **paper**: GraphGPT: Graph Instruction Tuning for Large Language Models
- **authors**: Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, Chao Huang (HKU Data Intelligence Lab + Baidu Inc.)
- **venue**: SIGIR 2024 (full paper track); arXiv:2310.13023
- **official_repository**: https://github.com/HKUDS/GraphGPT (verified, 832 stars, default branch `main`, Apache-2.0)
- **selected_commit**: `db25a66fd23b861156e6d7324f9ee8bc91c6ce7c` (2024-06-25, "Update README.md")
- **framework**: PyTorch + HuggingFace `transformers` (FastChat-derived serving stack) + DeepSpeed + PEFT + PyTorch Lightning (the `_light` scripts)
- **requirements** (`requirements.txt`, ~200 lines — critical pins, fetched):
  ```
  transformers==4.31.0      <-- HARD BLOCKER, see below
  tokenizers==0.13.3
  accelerate==0.21.0
  peft==0.4.0
  deepspeed==0.10.0
  flash-attn==1.0.4
  ray==2.6.1
  pydantic==1.10.9          (v1 API — FastAPI/gradio serving code depends on it)
  numpy==1.24.2
  sentencepiece==0.1.99
  torch (not pinned in the file; the stack implies torch 2.0.x)
  ```
- **dataset_support**: **NO fraud datasets whatsoever.** GraphGPT is built for `ogbn-arxiv`, `PubMed`, and `Cora` node classification + link prediction. It consumes:
  - a `graph_data_path` (`.pt` bundle of PyG graphs),
  - an instruction JSON (`instruct_ds`, e.g. HF dataset `Jiabin99/Arxiv-PubMed-mix-NC-LP`),
  - `./arxiv_ti_ab.json` (arxiv titles+abstracts) — **hard-coded as `--graph_content ./arxiv_ti_ab.json` in both tuning scripts**,
  - a pretrained graph transformer `pretra_gnn` (released as `Jiabin99/Arxiv-PubMed-GraphCLIP-GT`).
- **training_entrypoint** (two stages + projector extraction):
  ```
  # Stage 1: self-supervised graph matching instruction tuning
  bash scripts/tune_script/graphgpt_stage1.sh
    -> python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
         graphgpt/train/train_mem.py --model_name_or_path ${model_path} --version v1 \
         --data_path ${instruct_ds} --graph_content ./arxiv_ti_ab.json \
         --graph_data_path ${graph_data_path} --graph_tower ${pretra_gnn} \
         --tune_graph_mlp_adapter True --graph_select_layer -2 --use_graph_start_end \
         --bf16 True --num_train_epochs 3 --learning_rate 2e-3 --model_max_length 2048 ...
  # Extract the trained projector
  bash scripts/tune_script/extract_projector.sh   # -> scripts/extract_graph_projector.py
  # Stage 2: task-specific instruction tuning
  bash scripts/tune_script/graphgpt_stage2.sh
  ```
  A Lightning variant exists: `scripts/tune_script_light/graphgpt_stage{1,2}_lightning.sh`.
  The README fills `model_path=../vicuna-7b-v1.5-16k` for both stages.
- **evaluation_entrypoint**:
  ```
  bash scripts/eval_script/graphgpt_eval.sh
    -> python3.8 ./graphgpt/eval/run_graphgpt.py --model-name ${output_model} \
         --prompting_file ${datapath} --graph_data_path ${graph_data_path} \
         --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
  # then
  python scripts/eval_script/cal_metric_arxiv.py
  ```

## Base LLM: can it be swapped to Qwen3-8B?

**Assumed backbone: Vicuna-7B (v1.1 / v1.5 / v1.5-16k), i.e. LLaMA architecture.** Verified:
- README §3.1: *"`Vicuna`: Prepare our base model Vicuna, which is an instruction-tuned chatbot and base model in our implementation. Please download its weights here. We generally utilize v1.1 and v1.5 model with 7B parameters."*
- Scripts set `model_path=../vicuna-7b-v1.5-16k`.
- Released checkpoint `Jiabin99/GraphGPT-7B-mix-all` is "based on Vicuna-7B-v1.5".
- README also contains a step "To support vicuna base model".

**The architecture is hard-coded to LLaMA, not merely configured.** From `graphgpt/model/GraphLlama.py` (line numbers verified):
```python
23  from transformers import AutoConfig, AutoModelForCausalLM, \
24                           LlamaConfig, LlamaModel, LlamaForCausalLM, ...
43  class GraphLlamaConfig(LlamaConfig):
44      model_type = "GraphLlama"
77  class GraphLlamaModel(LlamaModel):
78      config_class = GraphLlamaConfig
278 class GraphLlamaForCausalLM(LlamaForCausalLM):
279     config_class = GraphLlamaConfig
434 AutoConfig.register("GraphLlama", GraphLlamaConfig)
435 AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)
```
GraphGPT **subclasses** `LlamaModel`/`LlamaForCausalLM` to splice graph tokens into `inputs_embeds`. Passing `--model_name_or_path Qwen/Qwen3-8B` will **not** work: `GraphLlamaConfig` inherits `LlamaConfig`, so loading Qwen3 weights into it either errors on `model_type: qwen3` or silently mismatches (Qwen3 has QK-norm and different GQA head counts). There is a `graphgpt/model/model_adapter.py` / `model_registry.py` (inherited from FastChat) but these only select *conversation templates*, not the modelling class.

**Effort to actually reach Qwen3-8B: HIGH.** You must author a parallel `GraphQwen.py` that subclasses `Qwen3Model`/`Qwen3ForCausalLM`, replicating the graph-token injection in `forward()` (~430 lines to port), register it, and adjust the `v1` Vicuna conversation template to Qwen3's chat template. On top of that, `transformers==4.31.0` **cannot load Qwen3 at all** (Qwen3 needs `transformers>=4.51.0`, otherwise `KeyError: 'qwen3'`), so the whole dependency stack must be upgraded — which breaks `flash-attn==1.0.4`, `pydantic==1.10.9` (v1 vs v2), and the `peft==0.4.0`/`deepspeed==0.10.0` pins.

**Conclusion: DGP's claim that GraphGPT was run "using official code" with a "Qwen3-8B backbone" cannot both be literally true.** One of the two was necessarily relaxed. Record this as a headline deviation.

- **adaptation_required** (VERY HIGH):
  1. Port the graph-token injection to a Qwen3 modelling class (or accept Vicuna-7B and note the backbone deviation — **strongly recommended for a faithful, cheap reproduction**).
  2. Upgrade `transformers` 4.31 → >=4.51 and repair the resulting `flash-attn`/`pydantic`/`peft`/`deepspeed` breakage.
  3. Build a fraud instruction corpus for YelpReviews/AmazonVideo in GraphGPT's format, and replace the hard-coded `--graph_content ./arxiv_ti_ab.json`.
  4. Pretrain a graph transformer via `text-graph-grounding/` on the fraud graphs (the released GT checkpoint is Arxiv/PubMed-only).
  5. Swap the multi-class arxiv metric script for binary AUC / F1-macro / AUPRC, including a probability score for AUC (GraphGPT emits free text).
  6. 4×A100 assumed by the scripts (`--nproc_per_node=4`).
- **deviations**: Backbone (Vicuna-7B vs Qwen3-8B), dependency stack, new dataset pipeline, new graph encoder, new metric extraction. All must be logged.
- **license**: Apache-2.0
- **reproducibility_status**: **DIFFICULT.** Well-maintained, well-documented repo (best README in this set), but no fraud support, a frozen 2023-era dependency stack, and a LLaMA-hard-coded model class that makes DGP's Qwen3-8B claim implausible as stated.
