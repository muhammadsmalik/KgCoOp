#!/usr/bin/env python3
"""Inspect domain-id assignments in a configured DG dataloader."""

import argparse
from argparse import Namespace
from typing import Iterable

import torch

from dassl.data import DataManager
from train import setup_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/trainers/CSDG/office_home_clipart_standard.yaml",
        help="Path to the method config file to load",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional override for DATASET.ROOT",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to sample batches from",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="How many batches to inspect",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides (same format as train.py)",
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    ns = Namespace(
        config_file=args.config,
        dataset_config_file="",
        dataset="",
        trainer="",
        source_domains=None,
        target_domains=None,
        transforms=None,
        backbone="",
        head="",
        seed=-1,
        output_dir="",
        opts=args.opts or [],
    )
    cfg = setup_cfg(ns)
    if args.dataset_root:
        cfg.defrost()
        cfg.DATASET.ROOT = args.dataset_root
        cfg.freeze()
    return cfg


def choose_loader(dm: DataManager, split: str) -> Iterable:
    if split == "train":
        return dm.train_loader_x
    if split == "val":
        if dm.val_loader is None:
            raise ValueError("Validation loader is not defined for this dataset")
        return dm.val_loader
    return dm.test_loader


def inspect_batches(loader: Iterable, num_batches: int) -> None:
    iterator = iter(loader)
    total = min(num_batches, len(loader))
    for batch_idx in range(total):
        batch = next(iterator)
        domain = batch.get("domain")
        if domain is None:
            print(f"Batch {batch_idx}: domain field not present")
            continue
        unique, counts = torch.unique(domain, sorted=True, return_counts=True)
        summary = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"Batch {batch_idx}: domain ids {list(summary.keys())}, counts {summary}")


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    dm = DataManager(cfg)
    loader = choose_loader(dm, args.split)
    print("Configured source domains:", list(cfg.DATASET.SOURCE_DOMAINS))
    print("Configured target domains:", list(cfg.DATASET.TARGET_DOMAINS))
    inspect_batches(loader, args.num_batches)


if __name__ == "__main__":
    main()
