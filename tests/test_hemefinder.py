#!/usr/bin/env python
"""Tests for `hemefinder` package."""
import json
import os.path as osp

import numpy as np

from hemefinder import hemefinder


def test_hemefinder():
    cwd = osp.join(osp.dirname(__file__))
    pdb_path = osp.join(cwd, '../data/7bc7.pdb')
    target = json.load(open(osp.join(cwd, '7bc7.json')))
    hemefinder(pdb_path, 'output')
    calculated = json.load(open(osp.join('output', '7bc7.json')))
    for (k, val), (k2, val2) in zip(target.items(), calculated.items()):
        assert k == k2
