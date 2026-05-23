"""
Evaluation utilities for association-dataset networks.

This module is designed to be usable:
- during training (pass an in-memory model + dataset), and
- out-of-the-box on trained networks (load your checkpoint/model and call here).

First implemented evaluation:
    Positive-pair retrieval on sampled anchor-positive pairs.

Protocol:
1) sample N (default: 10,000) anchor-positive pairs from the dataset,
2) embed anchors and positives,
3) compute cosine-similarity retrieval from anchors to all positives,
4) report top-1 / top-5 retrieval accuracy,
5) report a Kendall tau ranking score (query-wise, averaged).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent
_LIBS = _REPO_ROOT / "libs"
for _p in (_LIBS, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import ParticlePatchPairDataset
""" from models.conv import PatchConvNet
from models.mlp import PatchMLP
from nas_201_api import NASBench201API
from xautodl.config_utils import dict2config
from xautodl.models import get_cell_based_tiny_net """


@dataclass
class PositivePairRetrievalResult:
    num_pairs: int
    top1_acc: float
    top1_acc_std: float
    top5_acc: float
    top5_acc_std: float
    kendall_tau: float
    kendall_tau_std: float
    kendall_num_queries: int
    kendall_num_candidates: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_pairs": int(self.num_pairs),
            "top1_acc": float(self.top1_acc),
            "top1_acc_std": float(self.top1_acc_std),
            "top5_acc": float(self.top5_acc),
            "top5_acc_std": float(self.top5_acc_std),
            "kendall_tau": float(self.kendall_tau),
            "kendall_tau_std": float(self.kendall_tau_std),
            "kendall_num_queries": int(self.kendall_num_queries),
            "kendall_num_candidates": int(self.kendall_num_candidates),
        }


def _extract_embedding(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, (tuple, list)):
        if not model_out:
            raise RuntimeError("Model output tuple/list is empty; cannot extract embedding.")
        emb = model_out[0]
        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(
                f"Expected first model output to be Tensor, got: {type(emb).__name__}"
            )
        return emb
    raise RuntimeError(
        "Unsupported model output type for embedding extraction: "
        f"{type(model_out).__name__}"
    )


def _count_inversions(values: Sequence[int]) -> int:
    arr = list(values)
    tmp = [0] * len(arr)

    def _merge_sort(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = _merge_sort(lo, mid) + _merge_sort(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                j += 1
                inv += mid - i
            k += 1
        while i < mid:
            tmp[k] = arr[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = arr[j]
            j += 1
            k += 1
        arr[lo:hi] = tmp[lo:hi]
        return inv

    return _merge_sort(0, len(arr))


def _kendall_tau_from_orders(order_ref: torch.Tensor, order_pred: torch.Tensor) -> float:
    """
    Compute Kendall tau for two permutations over the same item IDs.
    """
    m = int(order_ref.numel())
    if m <= 1:
        return 0.0

    rank_ref = torch.empty(m, dtype=torch.long, device=order_ref.device)
    rank_ref[order_ref] = torch.arange(m, device=order_ref.device)
    perm = rank_ref[order_pred].cpu().tolist()
    inversions = _count_inversions(perm)
    denom = m * (m - 1) // 2
    if denom == 0:
        return 0.0
    # tau = (concordant - discordant) / total = 1 - 2*discordant/total
    return 1.0 - 2.0 * float(inversions) / float(denom)


def _sample_unique_track_indices(dataset: Any, num_samples: int, rng: random.Random) -> List[int]:
    tracks = getattr(dataset, "_tracks", None)
    if not isinstance(tracks, list):
        raise RuntimeError(
            "Dataset does not expose '_tracks'; unique-particle sampling is unavailable."
        )
    n_tracks = len(tracks)
    if n_tracks <= 0:
        return []
    if num_samples > n_tracks:
        raise ValueError(
            f"Requested num_pairs={num_samples}, but only {n_tracks} unique particle tracks "
            "are available in this split. Reduce --num-pairs."
        )
    return rng.sample(range(n_tracks), num_samples)


def _normalize_metric_names(metrics: Optional[Iterable[str]]) -> Set[str]:
    if metrics is None:
        return {"top1", "top5", "kendall_tau"}
    normalized: Set[str] = set()
    for raw in metrics:
        key = str(raw).strip().lower().replace("_", "-")
        if key in ("top1", "top-1"):
            normalized.add("top1")
        elif key in ("top5", "top-5"):
            normalized.add("top5")
        elif key in ("kendall", "kendall-tau", "kendalltau", "tau"):
            normalized.add("kendall_tau")
        else:
            raise ValueError(
                f"Unknown metric '{raw}'. Supported metrics: top-1, top-5, kendall-tau."
            )
    if not normalized:
        raise ValueError("metrics must not be empty.")
    return normalized


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = float(sum(values) / len(values))
    if len(values) <= 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / float(len(values))
    return m, float(var ** 0.5)


def gather_unique_particle_pairs(
    dataset: Any,
    track_indices: Sequence[int],
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
  Load anchor/positive patch tensors and L2-normalized embeddings for unique tracks.

  Returns ``(anchors, positives, emb_a, emb_p)`` each with shape ``[N, ...]``.
    """
    anchors, positives = _load_anchor_positive_tensors(dataset, track_indices)
    emb_a, emb_p = _embed_tensor_pairs(anchors, positives, model, device, batch_size)
    return anchors, positives, emb_a, emb_p


def compute_embedding_similarity_matrix(emb_a: torch.Tensor, emb_p: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix ``[N, N]`` with ``emb_a[i]`` vs ``emb_p[j]`` (both normalized)."""
    return emb_a @ emb_p.t()


def _load_anchor_positive_tensors(
    dataset: Any,
    track_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    anchors: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []

    tracks = getattr(dataset, "_tracks", None)
    sampler = getattr(dataset, "_sample_positive_indices", None)
    transform = getattr(dataset, "transform", None)
    if not isinstance(tracks, list) or sampler is None or transform is None:
        raise RuntimeError(
            "Dataset must expose _tracks, _sample_positive_indices, and transform."
        )

    split_dir = getattr(dataset, "split_dir", None)
    if split_dir is None:
        raise RuntimeError("Dataset must expose split_dir.")

    for ti in track_indices:
        track = tracks[int(ti)]
        i, j, _ = sampler(track)
        rels = track["paths_rel"]
        xa = transform(Image.open(split_dir / rels[i].replace("\\", "/")).convert("RGB"))
        xp = transform(Image.open(split_dir / rels[j].replace("\\", "/")).convert("RGB"))
        if not isinstance(xa, torch.Tensor) or not isinstance(xp, torch.Tensor):
            raise RuntimeError("Transform must return torch.Tensor.")
        anchors.append(xa)
        positives.append(xp)

    return torch.stack(anchors, dim=0), torch.stack(positives, dim=0)


def _embed_tensor_pairs(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    emb_a: List[torch.Tensor] = []
    emb_p: List[torch.Tensor] = []
    n = int(anchors.shape[0])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xa = anchors[start:end].to(device, non_blocking=True)
            xp = positives[start:end].to(device, non_blocking=True)
            za = F.normalize(_extract_embedding(model(xa)), dim=1)
            zp = F.normalize(_extract_embedding(model(xp)), dim=1)
            emb_a.append(za.detach().cpu())
            emb_p.append(zp.detach().cpu())
    return torch.cat(emb_a, dim=0), torch.cat(emb_p, dim=0)


def _gather_pairs_unique_particles(
    dataset: Any,
    track_indices: Sequence[int],
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, emb_a, emb_p = gather_unique_particle_pairs(
        dataset, track_indices, device, model, batch_size
    )
    return emb_a.to(device), emb_p.to(device)


def evaluate_positive_pair_retrieval(
    model: torch.nn.Module,
    dataset: Any,
    *,
    num_pairs: int = 10_000,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    seed: int = 0,
    kendall_num_queries: int = 256,
    kendall_num_candidates: int = 512,
    retrieval_chunk_size: int = 1024,
    metrics: Optional[Iterable[str]] = None,
    group_size: Optional[int] = None,
) -> PositivePairRetrievalResult:
    """
    Evaluate positive-pair retrieval.

    Args:
        model: Network that returns embedding tensor or (embedding, ...).
        dataset: Association dataset where ``dataset[i]`` yields at least
                 ``(anchor, positive, ...)`` tensors.
        num_pairs: Number of anchor-positive pairs to sample. Particles are unique,
                   i.e. the same ``(video_id, particle_id)`` track is never sampled
                   twice in one evaluation call.
        batch_size: Embedding forward batch size.
        device: Evaluation device. Defaults to model device if inferable.
        seed: Sampling seed.
        kendall_num_queries: Number of queries used for Kendall tau estimate.
        kendall_num_candidates: Candidate set size per query for Kendall tau.
        retrieval_chunk_size: Query chunk size for top-k retrieval computation.
        metrics: Metrics to compute. Supported names:
                 ``"top-1"``, ``"top-5"``, ``"kendall-tau"``.
                 Default (None): compute all.
        group_size: Retrieval group size. Retrieval/ranking is computed only within
                    each group, then averaged over groups. Default (None): use
                    ``num_pairs`` (single group over all samples).
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if retrieval_chunk_size <= 0:
        raise ValueError("retrieval_chunk_size must be > 0")
    if group_size is not None and int(group_size) <= 0:
        raise ValueError("group_size must be > 0 when provided")
    if group_size is not None and int(num_pairs) % int(group_size) != 0:
        raise ValueError(
            f"num_pairs ({int(num_pairs)}) must be divisible by group_size ({int(group_size)})."
        )
    selected = _normalize_metric_names(metrics)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(int(seed))
    sampled_track_indices = _sample_unique_track_indices(dataset, int(num_pairs), rng)
    if not sampled_track_indices:
        raise RuntimeError("No unique particle tracks drawn from dataset.")

    A, P = _gather_pairs_unique_particles(
        dataset, sampled_track_indices, device, model, int(batch_size)
    )
    n = int(A.shape[0])
    effective_group_size = n if group_size is None else int(group_size)
    effective_group_size = max(1, min(effective_group_size, n))

    top1_vals: List[float] = []
    top5_vals: List[float] = []
    tau_vals_all: List[float] = []
    total_kendall_queries = 0
    kendall_candidate_caps: List[int] = []

    for g_start in range(0, n, effective_group_size):
        g_end = min(g_start + effective_group_size, n)
        Ag = A[g_start:g_end]
        Pg = P[g_start:g_end]
        ng = int(Ag.shape[0])
        if ng <= 0:
            continue

        targets_g = torch.arange(ng, device=device)

        # Retrieval metrics (within group only).
        if "top1" in selected or "top5" in selected:
            top1_hits = 0
            top5_hits = 0
            k = min(5, ng)
            for start in range(0, ng, int(retrieval_chunk_size)):
                end = min(start + int(retrieval_chunk_size), ng)
                sims = Ag[start:end] @ Pg.t()
                topk = torch.topk(sims, k=k, dim=1, largest=True, sorted=True).indices
                t = targets_g[start:end].unsqueeze(1)
                if "top1" in selected:
                    top1_hits += int((topk[:, :1] == t).sum().item())
                if "top5" in selected:
                    top5_hits += int((topk == t).any(dim=1).sum().item())
            if "top1" in selected:
                top1_vals.append(float(top1_hits) / float(ng))
            if "top5" in selected:
                top5_vals.append(float(top5_hits) / float(ng))

        # Kendall tau metric (within group only).
        if "kendall_tau" in selected:
            q_count = max(1, min(int(kendall_num_queries), ng))
            query_ids = rng.sample(range(ng), q_count) if q_count < ng else list(range(ng))
            total_kendall_queries += len(query_ids)
            cand_cap = min(int(kendall_num_candidates), max(0, ng - 1))
            kendall_candidate_caps.append(cand_cap)

            for qi in query_ids:
                candidate_pool = [j for j in range(ng) if j != qi]
                if not candidate_pool:
                    continue
                c_count = min(int(kendall_num_candidates), len(candidate_pool))
                cand = (
                    rng.sample(candidate_pool, c_count)
                    if c_count < len(candidate_pool)
                    else candidate_pool
                )
                cand_t = torch.tensor(cand, dtype=torch.long, device=device)

                q = Ag[qi : qi + 1]  # [1, D]
                ref_scores = (q @ Ag[cand_t].t()).view(-1)
                pred_scores = (q @ Pg[cand_t].t()).view(-1)

                # ranks over 0..c_count-1 (local candidate IDs)
                order_ref = torch.argsort(ref_scores, descending=True)
                order_pred = torch.argsort(pred_scores, descending=True)
                tau_vals_all.append(_kendall_tau_from_orders(order_ref, order_pred))

    top1, top1_std = _mean_std(top1_vals)
    top5, top5_std = _mean_std(top5_vals)
    kendall, kendall_std = _mean_std(tau_vals_all)
    kendall_candidates_effective = (
        int(round(sum(kendall_candidate_caps) / len(kendall_candidate_caps)))
        if kendall_candidate_caps
        else 0
    )

    return PositivePairRetrievalResult(
        num_pairs=n,
        top1_acc=top1,
        top1_acc_std=top1_std,
        top5_acc=top5,
        top5_acc_std=top5_std,
        kendall_tau=kendall,
        kendall_tau_std=kendall_std,
        kendall_num_queries=total_kendall_queries,
        kendall_num_candidates=kendall_candidates_effective,
    )


__all__ = [
    "PositivePairRetrievalResult",
    "build_model_from_runfolder",
    "build_dataset_from_runfolder",
    "evaluate_positive_pair_retrieval",
    "gather_unique_particle_pairs",
    "compute_embedding_similarity_matrix",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_paths(raw: Optional[str], *, runfolder: Optional[Path] = None) -> List[Path]:
    out: List[Path] = []
    if not raw:
        return out
    p = Path(str(raw))
    out.append(p)
    # Common local fallback: paths stored relative to repo root.
    out.append(_REPO_ROOT / p)
    if runfolder is not None:
        out.append(runfolder / p)
    # Unique, keep order.
    uniq: List[Path] = []
    seen = set()
    for q in out:
        s = str(q)
        if s not in seen:
            seen.add(s)
            uniq.append(q)
    return uniq


def _collect_global_config_data_values(key: str) -> List[str]:
    vals: List[str] = []
    for cfg_path in sorted((_REPO_ROOT / "config" / "global").glob("*.json")):
        try:
            cfg = _load_json(cfg_path)
            v = cfg.get("data", {}).get(key, None)
            if v:
                vals.append(str(v))
        except Exception:
            continue
    # Unique, keep order.
    uniq: List[str] = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _resolve_data_path(
    *,
    key: str,
    preferred_value: Optional[str],
    runfolder: Optional[Path] = None,
    expect_dir: bool = False,
) -> str:
    checked: List[Path] = []
    candidates: List[Path] = []

    for c in _candidate_paths(preferred_value, runfolder=runfolder):
        candidates.append(c)

    for alt_raw in _collect_global_config_data_values(key):
        for c in _candidate_paths(alt_raw, runfolder=runfolder):
            candidates.append(c)

    # Unique while preserving order.
    uniq_candidates: List[Path] = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq_candidates.append(c)

    for c in uniq_candidates:
        checked.append(c)
        if expect_dir:
            if c.is_dir():
                return str(c)
        else:
            if c.is_file():
                return str(c)

    kind = "directory" if expect_dir else "file"
    checked_str = "\n  - ".join(str(p) for p in checked) if checked else "<none>"
    raise FileNotFoundError(
        f"Could not resolve existing {kind} for data.{key}. Checked:\n  - {checked_str}"
    )


def _resolve_latest_checkpoint(runfolder: Path, checkpoint_name: Optional[str]) -> Path:
    checkpoints_dir = runfolder / "checkpoints"
    if checkpoint_name:
        p = checkpoints_dir / f"latest.{checkpoint_name}.torch"
        if p.is_file():
            return p
    latest = sorted(checkpoints_dir.glob("latest.*.torch"))
    if latest:
        return latest[0]
    raise FileNotFoundError(f"No latest checkpoint found in: {checkpoints_dir}")


def _infer_backbone(experiment: Dict[str, Any], custom: Dict[str, Any]) -> str:
    raw = custom.get("model")
    if raw is None:
        exp_args = experiment.get("experiment_args", experiment)
        if isinstance(exp_args, dict):
            raw = exp_args.get("model")
    key = str(raw or "nb201").strip().lower().replace("-", "_")
    if key in ("conv", "cnn"):
        return "conv"
    if key == "mlp":
        return "mlp"
    if key in ("nb201", "nasbench201"):
        return "nb201"
    return key


def build_model_from_runfolder(
    runfolder: Path,
    arch_index_override: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load the trained network from an xpLog runfolder (NB-201, MLP, or Conv).

    Returns ``(model, settings, experiment, info)`` where ``info`` contains
    ``backbone`` and optionally ``arch_index``.
    """
    settings_path = runfolder / "settings.config"
    experiment_path = runfolder / "experiment.config"
    argparse_path = runfolder / "argparse.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Missing settings.config: {settings_path}")
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Missing experiment.config: {experiment_path}")

    settings = _load_json(settings_path)
    experiment = _load_json(experiment_path)

    file_checkpoints = settings.get("runfolder", {}).get("file_checkpoints", "chkpt")
    chk_path = _resolve_latest_checkpoint(runfolder, file_checkpoints)
    try:
        chk = torch.load(str(chk_path), map_location="cpu", weights_only=False)
    except TypeError:
        chk = torch.load(str(chk_path), map_location="cpu")

    custom = chk.get("custom", {})
    if not isinstance(custom, dict):
        custom = {}
    backbone = _infer_backbone(experiment, custom)

    if backbone == "mlp":
        cfg = dict(custom.get("mlp_config", {}))
        exp_args = experiment.get("experiment_args", {})
        if not cfg and isinstance(exp_args, dict):
            hidden = exp_args.get("mlp_hidden_sizes")
            if hidden is None and "mlp_hidden" in exp_args:
                hidden = [
                    int(p.strip())
                    for p in str(exp_args["mlp_hidden"]).split(",")
                    if p.strip()
                ]
            cfg = {
                "image_size": int(exp_args.get("image_size", 16)),
                "in_channels": 3,
                "hidden_sizes": tuple(int(x) for x in (hidden or [])),
                "embed_dim": int(exp_args.get("mlp_embed_dim", 128)),
                "dropout": float(exp_args.get("mlp_dropout", 0.0)),
            }
        model = PatchMLP(
            image_size=int(cfg.get("image_size", 16)),
            in_channels=int(cfg.get("in_channels", 3)),
            hidden_sizes=cfg.get("hidden_sizes", ()),
            embed_dim=int(cfg.get("embed_dim", 128)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
        info: Dict[str, Any] = {"backbone": "mlp", "checkpoint": str(chk_path)}
    elif backbone == "conv":
        cfg = dict(custom.get("conv_config", {}))
        exp_args = experiment.get("experiment_args", {})
        if not cfg and isinstance(exp_args, dict):
            cfg = {
                "in_channels": 3,
                "conv_channels": tuple(int(x) for x in exp_args.get("conv_channels", [32, 32])),
                "conv_kernel_sizes": tuple(
                    int(x) for x in exp_args.get("conv_kernel_sizes", [3, 3])
                ),
                "embed_dim": int(exp_args.get("conv_embed_dim", 128)),
            }
        model = PatchConvNet(
            in_channels=int(cfg.get("in_channels", 3)),
            conv_channels=cfg.get("conv_channels", (32, 32)),
            conv_kernel_sizes=cfg.get("conv_kernel_sizes", (3, 3)),
            embed_dim=int(cfg.get("embed_dim", 128)),
        )
        info = {"backbone": "conv", "checkpoint": str(chk_path)}
    else:
        arch_index: Optional[int] = arch_index_override
        if arch_index is None and argparse_path.is_file():
            parsed = _load_json(argparse_path)
            if "arch_index" in parsed:
                arch_index = int(parsed["arch_index"])
        if arch_index is None and "arch_index" in custom:
            arch_index = int(custom["arch_index"])
        if arch_index is None:
            raise RuntimeError(
                "Could not infer architecture index from argparse.json/checkpoint. "
                "Pass --arch-index explicitly."
            )

        nb201_api = _resolve_data_path(
            key="nb201_api",
            preferred_value=settings.get("data", {}).get("nb201_api", None),
            runfolder=runfolder,
            expect_dir=False,
        )
        api = NASBench201API(str(nb201_api), verbose=False)
        if arch_index < 0 or arch_index >= len(api.meta_archs):
            raise ValueError(
                f"arch-index must be in [0, {len(api.meta_archs) - 1}], got {arch_index}"
            )
        net_config = api.get_net_config(arch_index, "cifar10")
        model = get_cell_based_tiny_net(dict2config(net_config, None))
        info = {"backbone": "nb201", "arch_index": int(arch_index), "checkpoint": str(chk_path)}

    net_state = chk.get("networks", {}).get("net", None)
    if net_state is None:
        raise RuntimeError(f"Checkpoint missing networks.net state: {chk_path}")
    model.load_state_dict(net_state)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, settings, experiment, info


def _build_model_from_runfolder(
    runfolder: Path,
    arch_index_override: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], int]:
    model, settings, experiment, info = build_model_from_runfolder(
        runfolder, arch_index_override=arch_index_override, device=device
    )
    if info.get("backbone") != "nb201":
        raise RuntimeError(
            f"Runfolder uses backbone={info.get('backbone')!r}; use build_model_from_runfolder()."
        )
    return model, settings, experiment, int(info["arch_index"])


def build_dataset_from_runfolder(
    settings: Dict[str, Any],
    experiment: Dict[str, Any],
    split: str,
    num_pairs: int,
) -> ParticlePatchPairDataset:
    return _build_dataset_from_runfolder(settings, experiment, split, num_pairs)


def _build_dataset_from_runfolder(
    settings: Dict[str, Any],
    experiment: Dict[str, Any],
    split: str,
    num_pairs: int,
) -> ParticlePatchPairDataset:
    data_root = _resolve_data_path(
        key="trackopt_association",
        preferred_value=settings.get("data", {}).get("trackopt_association", None),
        runfolder=None,
        expect_dir=True,
    )

    cfg = experiment.get("experiment_args", {})
    image_size = int(cfg.get("image_size", 64))
    min_gap = int(cfg.get("min_gap", 1))
    max_gap_raw = cfg.get("max_gap", None)
    max_gap = float("inf") if max_gap_raw is None else float(max_gap_raw)
    force_rebuild = bool(cfg.get("force_rebuild_cache", False))

    return ParticlePatchPairDataset(
        association_root=str(data_root),
        split=split,
        length=max(1, int(num_pairs)),
        image_size=image_size,
        min_gap=min_gap,
        max_gap=max_gap,
        force_rebuild_cache=force_rebuild,
    )


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run positive-pair retrieval evaluation from a train runfolder."
    )
    p.add_argument("--runfolder", type=Path, required=True, help="Path to xpLog runfolder.")
    p.add_argument("--split", type=str, default="val", help="Dataset split (default: val).")
    p.add_argument("--num-pairs", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics", type=str, default="top-1,top-5,kendall-tau")
    p.add_argument("--kendall-num-queries", type=int, default=256)
    p.add_argument("--kendall-num-candidates", type=int, default=512)
    p.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "Restrict retrieval to groups of this size and average over groups. "
            "Default: num-pairs (single global group)."
        ),
    )
    p.add_argument("--retrieval-chunk-size", type=int, default=1024)
    p.add_argument("--arch-index", type=int, default=None, help="Optional override.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device, e.g. cpu / cuda / cuda:0 (default: auto).",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output json path (default: <runfolder>/evaluation/retrieval_metrics.json).",
    )
    return p.parse_args()


def _main() -> None:
    args = _parse_cli()
    runfolder = args.runfolder.resolve()
    if not runfolder.is_dir():
        raise FileNotFoundError(f"Runfolder not found: {runfolder}")

    device = torch.device(args.device) if args.device else None
    model, settings, experiment, arch_index = _build_model_from_runfolder(
        runfolder=runfolder,
        arch_index_override=args.arch_index,
        device=device,
    )
    if device is None:
        device = next(model.parameters()).device

    dataset = _build_dataset_from_runfolder(
        settings=settings,
        experiment=experiment,
        split=args.split,
        num_pairs=args.num_pairs,
    )
    metric_list = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    result = evaluate_positive_pair_retrieval(
        model=model,
        dataset=dataset,
        num_pairs=int(args.num_pairs),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed),
        metrics=metric_list,
        kendall_num_queries=int(args.kendall_num_queries),
        kendall_num_candidates=int(args.kendall_num_candidates),
        group_size=args.group_size,
        retrieval_chunk_size=int(args.retrieval_chunk_size),
    )

    payload = result.as_dict()
    payload.update(
        {
            "runfolder": str(runfolder),
            "split": str(args.split),
            "arch_index": int(arch_index),
            "device": str(device),
            "requested_metrics": metric_list,
            "group_size": (int(args.group_size) if args.group_size is not None else int(args.num_pairs)),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    out_json = args.save_json
    if out_json is None:
        out_json = runfolder / "evaluation" / "retrieval_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")
    print(f"Saved evaluation JSON: {out_json}")


if __name__ == "__main__":
    _main()

