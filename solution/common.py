"""Shared types and constants. One internal match format for every data source.

Everything downstream (features, decoder, benchmark) works on a Match, so the same code runs on
the external tuning games and on the competition match.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

H, A, D, X = 0, 1, 2, 3          # same codes as evaluation/possession_metrics.py
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25
EXTERNAL_DIR = os.path.join(ROOT, "external")
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR = os.path.join(HERE, "out")


@dataclass
class Match:
    name: str
    ball: np.ndarray                 # (n, 2) metres, NaN where the ball is not tracked
    pos: np.ndarray                  # (n, S, 2) metres, NaN where a slot is empty
    spd: np.ndarray                  # (n, S) m/s, NaN where a slot is empty
    team: np.ndarray                 # (S,) 0 = home, 1 = away
    blocks: List[Tuple[int, int]]    # half-open [start, stop) spans that are graded / decoded
    pitch: Tuple[float, float] = (105.0, 68.0)
    gt: Optional[np.ndarray] = None  # (n,) codes, X outside blocks. None for the competition match
    tags: Optional[list] = None      # Smart Tagging intervals, competition match only
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.ball)
