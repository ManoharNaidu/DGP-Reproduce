# HiGPT

- **paper**: HiGPT: Heterogeneous Graph Language Model
- **authors**: Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, Chao Huang (HKU Data Intelligence Lab + Baidu Inc.)
- **venue**: KDD 2024; arXiv:2402.16024
- **official_repository**: https://github.com/HKUDS/HiGPT (verified, 146 stars, default branch `main`, Apache-2.0)
- **selected_commit**: `2b0793e7bdfda693cbe84e5bd8632a59657c72a3` (2024-06-05, "Add files via upload")
- **framework**: PyTorch + HuggingFace `transformers` (FastChat-derived) + DeepSpeed + PEFT + PyTorch Lightning. Architecturally a heterogeneous-graph fork of GraphGPT — same file layout, same dependency stack.
- **requirements** (`requirements.txt` — **byte-identical key pins to GraphGPT**, fetched):
  ```
  transformers==4.31.0      <-- HARD BLOCKER for Qwen3, see below
  tokenizers==0.13.3
  accelerate==0.21.0
  peft==0.4.0
  deepspeed==0.10.0
  flash-attn==1.0.4
  ray==2.6.1
  pydantic==1.10.9
  numpy==1.24.2
  sentencepiece==0.1.99
  ```
  Repo also ships stale `__pycache__/*.cpython-38.pyc` files → **Python 3.8** is the intended interpreter (scripts literally invoke `python3.8`).
- **dataset_support**: **NO fraud datasets.** HiGPT targets heterogeneous academic/movie graphs: **IMDB, DBLP, ACM**. Evidence in-repo: `higpt/model/meta_hgt/meta_dict/{acm,dblp,imdb}/{node_type.pt,edge_type.pt}` — the meta node/edge type vocabularies are **shipped as pickled tensors per dataset**, so adding a new graph schema means generating new `node_type.pt` / `edge_type.pt`. Data fetched via `hi_datasets/get_stage1_data.sh` and `hi_datasets/get_stage2_data.sh`.
- **training_entrypoint** (two stages + graph tokenizer pretraining):
  ```
  # Heterogeneous graph tokenizer (MetaHGT) pretraining
  bash scripts/tune_script/run_graph_tokenizer.sh          # or run_graph_tokenizer_single.sh
  # Stage 1: heterogeneity-aware instruction tuning
  bash scripts/tune_script/higpt_stage_1.sh
    -> python3.8 -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
         higpt/train/train_hete_nopl.py --model_name_or_path ${base_model} --version v1 \
         --data_path ${data_path} --graph_root ./hi_datasets/matching_instruction \
         --graph_tower ${graph_tower} --tune_graph_mlp_adapter True --graph_select_layer -2 \
         --use_graph_start_end True --num_train_epochs 1 --learning_rate 2e-5 --model_max_length 2048 ...
  bash scripts/tune_script/extract_projector.sh
  # Stage 2: task-specific + in-context heterogeneous graph instruction tuning
  bash scripts/tune_script/higpt_stage_2.sh
  ```
  **`higpt_stage_1.sh` line 5 literally reads `base_model=/path/to/vicuna-7b-v1.5-16k`.**
- **evaluation_entrypoint**:
  ```
  bash scripts/eval_script/higpt_info_imdb_cot.sh
    -> python3.8 ./higpt/eval/run_higpt.py --model-name ${output_model}/higpt-stage2-imdb-metahgt-epoch15-mixcot-true-${num_shot} \
         --prompting_file ${datapath}/instruct_ds_imdb/.../IMDB_test_std_0_1000_cot_${cot_case}.json \
         --graph_root ${datapath} --output_res_path ... --start_id 0 --end_id 1000 --num_gpus 4
  # in-context variant: scripts/eval_script/hetegpt_info_imdb_cot_incontext.sh
  ```
  The eval script sweeps `num_shot_list=(1 3 5 10 20 40 60)` — HiGPT is evaluated few-shot, which may not match DGP's protocol.

## Base LLM: can it be swapped to Qwen3-8B?

**Assumed backbone: Vicuna-7B-v1.5-16k (LLaMA architecture)** — hard-coded as the default in `scripts/tune_script/higpt_stage_1.sh`.

**Same LLaMA hard-coding as GraphGPT.** From `higpt/model/HeteroLlama.py` (verified):
```python
23  ... LlamaConfig, LlamaModel, LlamaForCausalLM, ...
42  class HeteroLlamaConfig(LlamaConfig):
43      model_type = "HeteroLlama"
79  class HeteroLlamaModel(LlamaModel):
80      config_class = HeteroLlamaConfig
265 class HeteroLlamaForCausalLM(LlamaForCausalLM):
266     config_class = HeteroLlamaConfig
436 AutoConfig.register("HeteroLlama", HeteroLlamaConfig)
437 AutoModelForCausalLM.register(HeteroLlamaConfig, HeteroLlamaForCausalLM)
```
Plus the legacy `higpt/model/GraphLlama.py` with the same pattern. The repo does contain `higpt/model/chatglm_model.py` and `higpt/model/rwkv_model.py`, but those are **inherited FastChat serving adapters for the chat demo**, not alternative training backbones for the graph-token pathway — the graph-injection `forward()` exists only in the `*Llama.py` files.

**Effort to reach Qwen3-8B: HIGH — identical to GraphGPT.** You would write `HeteroQwen.py` subclassing `Qwen3Model`/`Qwen3ForCausalLM` and port ~440 lines of graph-token injection, then upgrade `transformers` 4.31 → >=4.51 (Qwen3 is unknown to 4.31: `KeyError: 'qwen3'`), which cascades into `flash-attn==1.0.4`, `pydantic` v1→v2, `peft==0.4.0`, `deepspeed==0.10.0` breakage, on a Python 3.8 codebase that modern `transformers` no longer supports (>=4.41 requires Python >=3.8 but the ecosystem has moved to 3.10+).

**Conclusion: as with GraphGPT, "official code" and "Qwen3-8B backbone" are mutually exclusive here.** Document whichever you choose.

- **adaptation_required** (VERY HIGH — the hardest baseline in the set):
  1. Port graph-token injection to a Qwen3 modelling class, **or** run Vicuna-7B-v1.5-16k and declare the backbone deviation (recommended).
  2. Upgrade the entire 2023-era dependency stack; abandon Python 3.8.
  3. **Define a heterogeneous schema for the fraud graphs.** YelpChi/Amazon are single-node-type multi-relation graphs — you must decide node/edge types and then generate new `node_type.pt` / `edge_type.pt` meta-dicts under `higpt/model/meta_hgt/meta_dict/<yourdataset>/`. DGP does not describe this choice.
  4. Pretrain the MetaHGT graph tokenizer on the fraud graphs (`HG_grounding/`, `run_graph_tokenizer.sh`).
  5. Build stage-1 (matching) and stage-2 (task + CoT + in-context) instruction corpora for fraud — HiGPT's stage-2 assumes mixed-CoT few-shot files that don't exist for Yelp/Amazon.
  6. Several scripts contain absolute Baidu-internal paths that must be fixed, e.g. `--graph_content /root/paddlejob/workspace/env_run/llm/GraphChat/playground/data/arxiv_ti_ab.json` and `--hetero_key_path /root/paddlejob/workspace/env_run/output/sample_instruct_ds/ann/hetero_key_order.json`, and `cd /path/to/HiGPT`.
  7. Binary AUC/F1-macro/AUPRC extraction from free-text output.
  8. 4×A100 assumed (`--nproc_per_node=4`, `--num_gpus 4`).
- **deviations**: Backbone, dependency stack, invented heterogeneous schema, new graph tokenizer, new instruction corpora, new metrics. The reproduced HiGPT number will be far from "official code".
- **license**: Apache-2.0
- **reproducibility_status**: **VERY DIFFICULT.** Leftover `.pyc`/`.DS_Store` files, hard-coded internal Baidu paths in the training scripts, per-dataset pickled type vocabularies, LLaMA-locked model class, and a frozen Python-3.8 stack.
