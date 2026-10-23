"""Datasets for decomposition and abstraction training.

Both datasets read from JSONL files (one JSON object per line).

  - data/decomposition.jsonl (1 line = 1 page)
  - data/abstraction.jsonl (1 line = 1 region)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from torch.utils.data import Dataset


class DecompositionDataset(Dataset):
    """Dataset for edge classifier training.

    Each sample is a page with serialized AXTree and region root backend IDs.
    Edge labels (cut/merge) are derived at load time from region_root_backend_ids.

    JSONL format per line:
    {
        "uid": "example_com_0",
        "url": "...",
        "axtree": "[68] RootWebArea ...\\n\\t[104] banner ...",
        "region_root_backend_ids": [104, 116, 106, ...]
    }
    """

    def __init__(self, data_path: str = "data/decomposition.jsonl") -> None:
        self.samples: List[Dict[str, Any]] = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class AbstractionDataset(Dataset):
    """Dataset for representor SFT training.

    Each sample is a region with subtree, purpose, and state_summary.

    JSONL format per line:
    {
        "uid": "example_com_0",
        "region_id": 0,
        "root_backend_id": 68,
        "subtree": "[68] RootWebArea ...",
        "purpose": "site navigation bar",
        "state_summary": "Contains links to ..."
    }
    """

    def __init__(self, data_path: str = "data/abstraction.jsonl") -> None:
        self.samples: List[Dict[str, Any]] = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
