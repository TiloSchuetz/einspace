import gzip
from os.path import join
from pickle import load

import numpy as np
import pandas as pd

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms

from einspace.utils import millify

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from pathlib import Path
import random
ImageSize = Union[int, Tuple[int, int]]


# --------------------------------------------------------
# NASBench360 imports
from einspace.data_utils.fsd50k import build_nasbench360_fsd_dataset
from einspace.data_utils.darcyflow import build_nasbench360_darcy_dataset
from einspace.data_utils.psicov import build_nasbench360_psicov_dataset
from einspace.data_utils.cosmic import build_nasbench360_cosmic_dataset
from einspace.data_utils.ecg import build_nasbench360_ecg_dataset
from einspace.data_utils.satellite import build_nasbench360_satellite_dataset
from einspace.data_utils.deepsea import build_nasbench360_deepsea_dataset
# --------------------------------------------------------


nas360_cfg = dict(
    fsd_root = "data/fsd50k",
    darcy_root = "data/darcyflow/",
    psicov_root = "data/psicov",
    cosmic_root = "data/cosmic/",
    ecg_root = "data/ecg",
    satellite_root = "data/satellite",
    deepsea_root = "data/deepsea"
)

unseen_datasets = [
    "addnist",
    "language",
    "multnist",
    "cifartile",
    "gutenberg",
    "isabella",
    "geoclassing",
    "chesseract",
    "cifar-10",
]

# particle tracking dataset

class ParticlePatchPairDataset(Dataset):
    """
    Triplet samples for particle re-identification from patch folders produced by
    ``create_particle_patch_dataset``:

        association_root / split / video_XXXXXX / particle_YYYYYY / frame_*_ann_*.png

    Each ``(video_*, particle_*)`` directory is one particle identity. Each sample
    returns an anchor patch, a positive from the **same** identity (different
    frame), and a negative from **another** identity. Filenames are parsed for
    ``frame_<image_id>_ann_<id>.png`` to measure temporal gap.

    Sampling constraints:
        Only anchor-positive pairs with ``min_gap <= gap <= max_gap`` are sampled.
        Valid index pairs are **not** stored (that can require tens of GiB); each
        ``__getitem__`` draws a random eligible ``(i, j)`` by rejection sampling with
        a deterministic fallback scan.

    Cache:
        Track metadata is cached at
        ``association_root/particle_patch_pair_cache_<split>.torch``.
        Set ``force_rebuild_cache=True`` to ignore and rebuild this cache.

    For each particle track, the largest observed frame gap is stored in
    ``max_gap_by_track[(video_id, particle_id)]``. The global maximum over all
    tracks remains available via ``max_gap``.

    Returns:
        anchor (Tensor): ``C×H×W``
        positive (Tensor): same
        negative (Tensor): same
        gap_frames (int): absolute frame distance between anchor and positive
        gap_max (int): maximum observed anchor-positive frame gap for that track
    """

    _FRAME_RE = re.compile(r"frame_(\d+)_ann_(\d+)\.png$", re.IGNORECASE)
    _CACHE_VERSION = 1

    def __init__(
        self,
        association_root: str,
        split: str,
        length: int = 65536,
        image_size: ImageSize = 32,
        min_gap: int = 1,
        max_gap: float = 3.0, #float("inf"), Tilo: align with Steffen's setup
        force_rebuild_cache: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(association_root)
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Association split not found: {self.split_dir}")

        if isinstance(image_size, int):
            size_hw = (image_size, image_size)
        else:
            size_hw = (int(image_size[0]), int(image_size[1]))

        if transform is None:
            from torchvision import transforms

            self.transform = transforms.Compose(
                [transforms.Resize(size_hw), transforms.ToTensor()]
            )
        else:
            self.transform = transform

        self.length = int(length)
        self.min_gap = int(min_gap)
        self.allowed_max_gap = float(max_gap)
        self.force_rebuild_cache = bool(force_rebuild_cache)
        if self.min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {self.min_gap}")
        if self.allowed_max_gap < float(self.min_gap):
            raise ValueError(
                f"max_gap must be >= min_gap ({self.min_gap}), got {self.allowed_max_gap}"
            )

        self._tracks: List[Dict[str, Any]] = []
        self.max_gap_observed = 1
        self.max_gap_by_track: Dict[Tuple[int, int], int] = {}

        cache_path = self.root / f"particle_patch_pair_cache_{self.split}.torch"
        raw_tracks = self._load_or_build_track_cache(
            cache_path, force_rebuild=self.force_rebuild_cache
        )

        for track in raw_tracks:
            video_id = int(track["video_id"])
            particle_id = int(track["particle_id"])
            frame_ids = [int(x) for x in track["frames"]]
            max_local = int(track["max_gap"])
            if max_local > self.max_gap_observed:
                self.max_gap_observed = max_local
            self.max_gap_by_track[(video_id, particle_id)] = max_local

            if not self._track_has_valid_positive_pair(frame_ids):
                continue

            self._tracks.append(
                {
                    "paths_rel": [str(rel).replace("\\", "/") for rel in track["paths_rel"]],
                    "frames": frame_ids,
                    "video_id": video_id,
                    "particle_id": particle_id,
                    "max_gap": max_local,
                }
            )

        if not self._tracks:
            raise RuntimeError(
                f"No tracks with valid positive pairs under {self.split_dir} "
                f"for min_gap={self.min_gap}, max_gap={self.allowed_max_gap}"
            )
        if len(self._tracks) < 2:
            raise RuntimeError(
                f"Need at least two distinct particle tracks (with 2+ patches each) "
                f"under {self.split_dir}; found only one."
            )

        self.max_gap_observed = max(self.max_gap_observed, 1)

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    @property
    def max_gap(self) -> int:
        """Backward-compatible alias for the observed global maximum track gap."""
        return self.max_gap_observed

    @classmethod
    def _parse_frame_id(cls, path: Path) -> int:
        m = cls._FRAME_RE.search(path.name)
        if m is None:
            return 0
        return int(m.group(1))

    def __len__(self) -> int:
        return self.length

    def _track_has_valid_positive_pair(self, frame_ids: List[int]) -> bool:
        """Whether any patch pair (i < j) satisfies ``min_gap <= |Δframe| <= max_gap``."""
        n = len(frame_ids)
        for i in range(n):
            fi = frame_ids[i]
            for j in range(i + 1, n):
                g = abs(fi - frame_ids[j])
                if self.min_gap <= g <= self.allowed_max_gap:
                    return True
        return False

    def _scan_first_valid_pair(self, frame_ids: List[int]) -> Tuple[int, int, int]:
        """First (i, j, gap) with i < j in gap range; used if rejection sampling fails."""
        n = len(frame_ids)
        for i in range(n):
            fi = frame_ids[i]
            for j in range(i + 1, n):
                g = abs(fi - frame_ids[j])
                if self.min_gap <= g <= self.allowed_max_gap:
                    return i, j, g
        raise RuntimeError(
            "internal: track has no valid positive pair despite init filter; "
            f"frames={n}"
        )

    def _sample_positive_indices(self, track: Dict[str, Any]) -> Tuple[int, int, int]:
        """Random (i, j, gap) with i < j and gap in ``[min_gap, max_gap]``."""
        frames: List[int] = track["frames"]
        n = len(frames)
        if n < 2:
            raise RuntimeError("internal: track needs at least two frames")
        max_tries = max(512, n * n)
        for _ in range(max_tries):
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            g = abs(frames[i] - frames[j])
            if self.min_gap <= g <= self.allowed_max_gap:
                return i, j, g
        return self._scan_first_valid_pair(frames)

    def _load_or_build_track_cache(
        self, cache_path: Path, force_rebuild: bool = False
    ) -> List[Dict[str, Any]]:
        if cache_path.is_file() and not force_rebuild:
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(cached, dict)
                    and cached.get("version") == self._CACHE_VERSION
                    and cached.get("split") == self.split
                    and isinstance(cached.get("tracks"), list)
                ):
                    return cached["tracks"]
            except Exception:
                pass

        tracks = self._build_track_index()
        payload = {
            "version": self._CACHE_VERSION,
            "split": self.split,
            "tracks": tracks,
        }
        try:
            torch.save(payload, cache_path)
        except Exception:
            # Caching is optional; dataset construction should still proceed.
            pass
        return tracks

    def _build_track_index(self) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        for video_dir in sorted(self.split_dir.glob("video_*")):
            vparts = video_dir.name.split("_")
            if len(vparts) < 2:
                continue
            try:
                video_id = int(vparts[1])
            except ValueError:
                continue

            for particle_dir in sorted(video_dir.glob("particle_*")):
                pparts = particle_dir.name.split("_")
                if len(pparts) < 2:
                    continue
                try:
                    particle_id = int(pparts[1])
                except ValueError:
                    continue

                pngs = sorted(particle_dir.glob("*.png"))
                if len(pngs) < 2:
                    continue

                frame_ids = [self._parse_frame_id(p) for p in pngs]
                max_local = 1
                for i in range(len(frame_ids)):
                    for j in range(i + 1, len(frame_ids)):
                        g = abs(frame_ids[i] - frame_ids[j])
                        if g > max_local:
                            max_local = g

                paths_rel = [
                    str(path.relative_to(self.split_dir)).replace("\\", "/")
                    for path in pngs
                ]
                tracks.append(
                    {
                        "video_id": video_id,
                        "particle_id": particle_id,
                        "frames": frame_ids,
                        "max_gap": max_local,
                        "paths_rel": paths_rel,
                    }
                )
        return tracks

    def __getitem__(self, index: int):
        del index
        track = random.choice(self._tracks)
        paths_rel = track["paths_rel"]
        i, j, gap = self._sample_positive_indices(track)

        neg_track = track
        for _ in range(50):
            cand = random.choice(self._tracks)
            if (
                cand["video_id"] != track["video_id"]
                or cand["particle_id"] != track["particle_id"]
            ):
                neg_track = cand
                break

        neg_rel = random.choice(neg_track["paths_rel"])

        def load_rel(rel: str):
            p = self.split_dir / rel.replace("\\", "/")
            im = Image.open(p).convert("RGB")
            return self.transform(im)

        return (
            load_rel(paths_rel[i]),
            load_rel(paths_rel[j]),
            load_rel(neg_rel),
            int(gap),
            int(track["max_gap"]),
        )

class CSAWM(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, loss_type="one_hot"):
        load_split = {"train": "train", "val": "train", "trainval": "train", "test": "test"}[split]
        self.info = pd.read_csv(join(root, "csawm", "labels", f"CSAW-M_{load_split}.csv"), header=0, delimiter=";")
        val_filenames = [
            line.replace("\n", "") for line in open(
                join(root, "csawm", "cross_validation", "CSAW-M_cross_validation_split1.txt"),
                "r"
            ).readlines()
        ]
        self.data, self.targets = [], []
        for _, row in self.info.iterrows():
            path = join(root, "csawm", "images", "preprocessed", load_split, row["Filename"])
            img = default_loader(path)
            if (
                (split == "train" and row["Filename"] not in val_filenames) or
                (split == "val" and row["Filename"] in val_filenames) or
                split in ["trainval", "test"]
            ):
                self.data.append(img)
                self.targets.append(row["Label"] - 1)
        self.transform = transform
        self.loss_type = loss_type

    def make_multi_hot(self, label, n_labels=8):
        multi_hot = [0] * (n_labels - 1)
        if label > 0:
            for i in range(label):
                multi_hot[i] = 1
        return torch.tensor(multi_hot, dtype=torch.float32)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.loss_type == "multi_hot":
            target = self.make_multi_hot(target)
        return img, target

    def __len__(self):
        return len(self.data)


class UnseenDataset(Dataset):
    def __init__(
        self, root, dataset, split="train", transform=None, image_size=None
    ):
        if split == "val":
            split = "valid"
        self.data = torch.tensor(
            np.load(
                join(root, dataset, f"{split}_x.npy"), allow_pickle=True
            ).astype(np.float32)
        )
        self.targets = torch.tensor(
            np.load(
                join(root, dataset, f"{split}_y.npy"), allow_pickle=True
            ).astype(int)
        )

        self.transform = transform
        # example transform
        if split == "train":
            self.mean = torch.mean(self.data, [0, 2, 3])
            self.std = torch.std(self.data, [0, 2, 3])
            transform = [
                transforms.Normalize(self.mean, self.std),
            ]
            if dataset == "chesseract":
                transform.append(
                    transforms.Pad(5, fill=0, padding_mode="constant")
                )
            transform.append(transforms.Resize(image_size))
            self.transform = transforms.Compose(transform)

        self.data = torch.stack([self.transform(img) for img in self.data])

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(datasets.CIFAR100):
    """
    Class that inherits from datasets.CIFAR100.
    It loads CIFAR100 and returns train_dataset, valid_dataset, and test_dataset
    Using indices from cifar100_train.indices and cifar100_valid.indices
    """
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        super().__init__(root=join(root, "cifar100"), train=split in ["train", "val"], transform=transform, target_transform=target_transform, download=download)
        if split == "train":
            self.indices = torch.load(f'{root}/cifar100/cifar100_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/cifar100/cifar100_valid.indices')
        elif split == "test":
            self.indices = torch.arange(len(self.data))
        self.data = self.data[self.indices]
        self.targets = torch.tensor(self.targets)[self.indices]


class CIFAR10(datasets.CIFAR10):
    """
    Class that inherits from datasets.CIFAR10.
    It loads CIFAR10 and returns train_dataset, valid_dataset, and test_dataset
    Using indices from cifar10_train.indices and cifar10_valid.indices
    """
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        super().__init__(root=join(root, "cifar10"), train=split in ["train", "val"], transform=transform, target_transform=target_transform, download=download)
        if split == "train":
            self.indices = torch.load(f'{root}/cifar10/cifar10_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/cifar10/cifar10_valid.indices')
        elif split == "test":
            self.indices = torch.arange(len(self.data))
        self.data = self.data[self.indices]
        self.targets = torch.tensor(self.targets)[self.indices]


class NinaPro(Dataset):
    """
    Class that loads the NinaPro dataset.
    18 classes, input shape (16, 52).
    """
    def __init__(self, root, split="train", transform=None):
        self.data = np.load(
            join(root, "ninapro", f"ninapro_{split}.npy"), allow_pickle=True
        ).astype(np.float32)
        self.targets = np.load(
            join(root, "ninapro", f"label_{split}.npy"), allow_pickle=True
        ).astype(int)
        self.data = torch.tensor(self.data)
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class Spherical(Dataset):
    """
    Version of CIFAR100 where each image has been projected onto a spherical surface.
    100 classes, 600 images per class, 60,000 images in total.
    """
    def __init__(self, root, split="train", transform=None):
        load_data = load(
            gzip.open(join(root, "spherical", "s2_cifar100.gz"), "rb")
        )
        # load the indices for the train and valid splits
        if split == "train":
            self.indices = torch.load(f'{root}/spherical/spherical_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/spherical/spherical_valid.indices')
        else:
            self.indices = torch.arange(len(load_data["test"]["images"]))
        # load the data and targets
        if split in ["train", "val"]:
            self.data = load_data["train"]["images"][self.indices]
            self.targets = np.array(load_data["train"]["labels"])[self.indices]
        else:
            self.data = load_data["test"]["images"][self.indices]
            self.targets = np.array(load_data["test"]["labels"])[self.indices]
        # transpose data to (N, H, W, C)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def get_data_loaders(
    dataset,
    batch_size,
    image_size,
    root="data",
    load_in_gpu=True,
    device=None,
    log=False,
):
    """Get data loaders for a given dataset."""
    trainvalset = None
    if dataset == "csawm":
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(**{'brightness': 0.2, 'contrast': 0.2}),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        trainset = CSAWM(root, "train", transform=train_transform, loss_type="multi_hot")
        valset = CSAWM(root, "val", transform=test_transform, loss_type="multi_hot")
        trainvalset = CSAWM(root, "trainval", transform=train_transform, loss_type="multi_hot")
        testset = CSAWM(root, "test", transform=test_transform, loss_type="multi_hot")
    elif dataset == "particle":
        trainset = ParticlePatchPairDataset(root + "/" + dataset, "train")
        valset = ParticlePatchPairDataset(root + "/" + dataset, "val")
        # did not inlcude trainval set
        testset = ParticlePatchPairDataset(root + "/" + dataset, "test")
    elif dataset in unseen_datasets:
        trainset = UnseenDataset(
            root, dataset, split="train", transform=None, image_size=image_size
        )
        valset = UnseenDataset(
            root, dataset, split="val", transform=trainset.transform
        )
        testset = UnseenDataset(
            root, dataset, split="test", transform=trainset.transform
        )
    elif dataset == "mnist":
        dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            download=True,
        )
        testset = datasets.MNIST(
            root=root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            download=True,
        )
        trainset, valset = random_split(
            dataset,
            [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
        )
    elif dataset == "cifar10":
        trainset = CIFAR10(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
        valset = CIFAR10(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
        testset = CIFAR10(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
    elif dataset == "cifar100":
        trainset = CIFAR100(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
        valset = CIFAR100(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
        testset = CIFAR100(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
    elif dataset == "spherical":
        trainset = Spherical(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        valset = Spherical(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        testset = Spherical(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
    elif dataset == "ninapro":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        trainset = NinaPro(root, split="train", transform=transform)
        valset = NinaPro(root, split="val", transform=transform)
        testset = NinaPro(root, split="test", transform=transform)
    elif dataset == "fsd50k":
        trainset    = build_nasbench360_fsd_dataset("train", nas360_cfg)
        valset      = build_nasbench360_fsd_dataset("val", nas360_cfg)
        trainvalset = build_nasbench360_fsd_dataset("trainval", nas360_cfg)
        testset     = build_nasbench360_fsd_dataset("test", nas360_cfg)
    elif dataset == "darcyflow":
        trainset, valset, testset, y_normalizer = build_nasbench360_darcy_dataset(nas360_cfg)
    elif dataset == "psicov":
        (
            trainset, valset, 
            testset, test_my_list, test_length_dict
        ) = build_nasbench360_psicov_dataset(nas360_cfg)
    elif dataset == "cosmic":
        trainset, valset, testset = build_nasbench360_cosmic_dataset(nas360_cfg)
    elif dataset == "ecg":
        trainset, valset, testset = build_nasbench360_ecg_dataset(nas360_cfg)
    elif dataset == "satellite":
        trainset, valset, testset = build_nasbench360_satellite_dataset(nas360_cfg)
    elif dataset == "deepsea":
        trainset, valset, testset = build_nasbench360_deepsea_dataset(nas360_cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # load data in GPU
    if load_in_gpu:
        try:
            trainset.data = trainset.data.to(device)
            valset.data = valset.data.to(device)
            testset.data = testset.data.to(device)
            trainset.targets = trainset.targets.to(device)
            valset.targets = valset.targets.to(device)
            testset.targets = testset.targets.to(device)
            # report how much GPU memory is used
            element_size = trainset.data.element_size()
            nelement = (
                trainset.data.nelement()
                + valset.data.nelement()
                + testset.data.nelement()
            )
            size = element_size * nelement
            print(
                f"Loaded {dataset} in GPU. Size: {millify(size, bytes=True)}"
            )
        except Exception as e:
            print(f"Tried moving {dataset} to GPU memory, but failed.")
            print(f"\t{e}")

    pin_memory = not load_in_gpu
    num_workers = 0 if load_in_gpu else 12 #Tilo align with Steffen's setup
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
    )
    if dataset in ["fsd50k"]:
        val_loader = valset
    else:
        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )
    if trainvalset is None:
        trainvalset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    trainval_loader = DataLoader(
        trainvalset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
    )
    if dataset in ["fsd50k"]:
        test_loader = testset
    else:
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loader, val_loader, trainval_loader, test_loader
