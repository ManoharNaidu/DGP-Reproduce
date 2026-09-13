"""Run the OFFICIAL ConsisGAD code (Xtra-Computing/ConsisGAD @ 36811c5b) on a DGP graph and split.

Provenance: BASELINE_ADAPTED. The upstream algorithm, model and training loop are used unmodified; upstream files are
never edited. This adapter only replaces the data entry point and captures predictions:

  1. `modules.data_loader.get_index_loader_test` is replaced by a loader that builds the DGL heterograph from our
     exported bundle and uses OUR train/val/test ids instead of ConsisGAD's own random split. The loaders it returns
     are built exactly as upstream builds them (same batch sizes, same labeled/unlabeled construction: all
     train+val+test nodes feed the unlabeled consistency loader, as in upstream; no labels are exposed through it).
  2. `main.get_model_pred` is wrapped to record each epoch's validation and test probabilities, so the test
     predictions at upstream's model-selection point (best validation AUROC, strict '>') can be saved for
     scripts/evaluate.py to recompute metrics with the shared protocol.
  3. Environment fixes that do not touch the method: `wandb` (imported, never used) is stubbed; the module-level
     `args` global that upstream's SoftAttentionDrop reads when run as a script is injected; `store-model` is forced
     off so upstream's shipped `model-weights/amazon.pth` is never overwritten.

Runs INSIDE the ConsisGAD environment (python 3.9, torch 1.13.1, dgl 1.1.0), so it imports nothing from dgp_repro:

    <env-python> src/dgp_repro/baselines/consisgad_adapter.py --bundle datasets/processed/amazonvideo/baseline_bundle.npz \
        --config methods/consisgad/upstream/config/amazon.yml --seeds 0 1 2 3 4 --out results/raw
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "methods" / "consisgad" / "upstream"
PINNED_COMMIT = "36811c5b"


def load_bundle(path):
    import dgl
    import torch

    z = np.load(path, allow_pickle=False)
    relations = [str(r) for r in z["relations"]]
    graph = dgl.heterograph({("review", rel, "review"): (torch.as_tensor(z[f"src_{rel}"]), torch.as_tensor(z[f"dst_{rel}"]))
                             for rel in relations}, num_nodes_dict={"review": int(z["num_nodes"])})
    graph.ndata["feature"] = torch.as_tensor(z["features"], dtype=torch.float32)
    graph.ndata["label"] = torch.as_tensor(z["labels"], dtype=torch.long)
    split = {name: z[name].astype(np.int64) for name in ("train", "val", "test")}
    return graph, split, str(z["dataset"])


def make_loader_factory(graph, split):
    """Mirror of upstream get_index_loader_test, with our split instead of train_test_split."""
    import torch
    from torch.utils.data import DataLoader as torch_dataloader

    def get_index_loader_dgp(name, batch_size, unlabel_ratio=1, training_ratio=-1, shuffle_train=True, to_homo=False):
        labels = graph.ndata["label"]
        for mask_name, ids in (("train_mask", split["train"]), ("val_mask", split["val"]), ("test_mask", split["test"])):
            mask = torch.zeros_like(labels).bool()
            mask[ids] = True
            graph.ndata[mask_name] = mask
        train_nids, valid_nids, test_nids = split["train"], split["val"], split["test"]
        unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])  # as upstream
        power = 16  # upstream value for non-tfinance datasets
        valid_loader = torch_dataloader(valid_nids, batch_size=2 ** power, shuffle=False, drop_last=False, num_workers=0)
        test_loader = torch_dataloader(test_nids, batch_size=2 ** power, shuffle=False, drop_last=False, num_workers=0)
        labeled_loader = torch_dataloader(train_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
        unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train,
                                            drop_last=True, num_workers=0)
        return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader

    return get_index_loader_dgp


def run(bundle, config_path, seeds, out_dir, epochs=None):
    sys.path.insert(0, str(UPSTREAM))
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    os.chdir(UPSTREAM)  # upstream uses relative paths
    import dgl
    import torch
    import yaml

    import main as upstream_main  # noqa: E402  (upstream ConsisGAD main.py)

    graph, split, dataset = load_bundle(bundle)
    upstream_main.get_index_loader_test = make_loader_factory(graph, split)

    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["store-model"] = False
    if epochs is not None:
        cfg["epochs"] = epochs
    cfg["device"] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    labels = graph.ndata["label"].numpy()
    original_get_pred = upstream_main.get_model_pred
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        args = dict(cfg)
        upstream_main.args = args  # the global upstream's SoftAttentionDrop reads when run as a script
        epoch_preds, started = [], time.time()

        def recording_get_pred(model, g, loader, sampler, a, _store=epoch_preds):
            pred, target = original_get_pred(model, g, loader, sampler, a)
            _store.append(pred.detach().cpu().numpy().copy())
            return pred, target

        upstream_main.get_model_pred = recording_get_pred
        upstream_main.run_model(args)  # upstream training loop, unchanged

        from sklearn.metrics import roc_auc_score
        val_preds, test_preds = epoch_preds[0::2], epoch_preds[1::2]  # val_epoch calls val then test
        best, best_auc = 0, -1.0
        for e, vp in enumerate(val_preds):
            auc = roc_auc_score(labels[split["val"]], vp)
            if auc > best_auc:  # strict '>' as upstream
                best, best_auc = e, auc

        run_id = f"consisgad_{dataset}_seed{seed}_{PINNED_COMMIT}"
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "split", "label", "p_fraud"])
            for name, prob in (("val", val_preds[best]), ("test", test_preds[best])):
                for node, p in zip(split[name], prob):
                    w.writerow([int(node), name, int(labels[node]), f"{float(p):.6f}"])
        manifest = {
            "run_id": run_id, "experiment": f"consisgad_{dataset}", "seed": seed, "mode": "CPU_REPRODUCTION"
            if not torch.cuda.is_available() else "GPU_REPRODUCTION",
            "method": "ConsisGAD", "provenance": "BASELINE_ADAPTED",
            "upstream": {"repo": "https://github.com/Xtra-Computing/ConsisGAD", "commit": PINNED_COMMIT, "config": str(config_path)},
            "config": {k: (str(v) if k == "device" else v) for k, v in cfg.items()},
            "selected_epoch": best, "epochs_run": len(val_preds), "runtime_seconds": round(time.time() - started, 1),
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__},
            "dataset_summary": {"name": dataset, "nodes": int(graph.num_nodes())},
            "split_sizes": {k: int(len(v)) for k, v in split.items()},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"seed {seed}: best epoch {best}, val AUROC {100 * best_auc:.2f}, wrote {run_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", default=str(UPSTREAM / "config" / "amazon.yml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=str(REPO / "results" / "raw"))
    p.add_argument("--epochs", type=int, help="override epochs (smoke checks only)")
    a = p.parse_args()
    run(Path(a.bundle).resolve(), Path(a.config).resolve(), a.seeds, Path(a.out).resolve(), a.epochs)
