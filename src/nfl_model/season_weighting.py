"""Current-season emphasis; v7 player multipliers are a reviewed user preference.

The stronger player weights trade stability for responsiveness. Historical
component tests do not establish an accuracy improvement over the v6 policy.
"""
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
