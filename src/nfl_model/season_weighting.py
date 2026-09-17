"""Multipliers selected on historical season transitions, not live results."""
import json
from pathlib import Path
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def policy():
    return json.loads(Path(__file__).with_suffix('.json').read_text())


def weights(frame,season,component,recency=True):
    result=np.arange(1,len(frame)+1,dtype=float) if recency else np.ones(len(frame))
    return result*np.where(frame.season.eq(season),policy()[component],1.)
