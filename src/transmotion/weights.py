"""
Module for loading and saving module weights. 


Key use cases:

- Port pretrained weights from original repo to this one

"""

from typing import Dict
import numpy as np


def param_name_iou(left_name: str, right_name: str) -> float:
    lhs = set(left_name.split("."))
    rhs = set(right_name.split("."))
    intersection = len(lhs.intersection(rhs))
    union_ = len(lhs.union(rhs))
    return intersection / union_


def reconcile_weights(orig_state_dict: Dict, new_state_dict: Dict):
    # classify params by their shapes (those have to match)
    shapes = {}

    def _add_shape(k, v, model):
        shape = tuple(v.shape)

        if shape not in shapes:
            shapes[shape] = {"old": [], "new": []}

        shapes[shape][model].append(k)

    for k, v in orig_state_dict.items():
        _add_shape(k, v, "old")

    for k, v in new_state_dict.items():
        _add_shape(k, v, "new")

    # more than one param name per shape? now we need to
    conflicts = {}
    mappings = {}
    for k, v in shapes.items():
        h, w = len(v["old"]), len(v["new"])

        if h != w:
            raise ValueError(
                f"Not same number of params with {k} shape: Old: {v['old']} New: {v['new']}"
            )

        if h > 1:
            # find intersectio over union of param names.
            conflicts[k] = np.zeros((h, w), dtype=np.float32)

            for ridx, old_par in enumerate(v["new"]):
                for cidx, new_par in enumerate(v["old"]):
                    conflicts[k][ridx, cidx] = param_name_iou(old_par, new_par)

            # if the best matches are diagonal we can asusme that params are at oreder
            # (we didn't change te order of params in the modules, just some names)
            best_fits = conflicts[k].argmax(axis=1)
            if not (best_fits == np.arange(h)).all():
                # if this condtioin is broken. it might be an issue with the measurement.
                # we compare argmax to digaonal, digaonal must be >= argmax if the order is correct
                if not (
                    conflicts[k][np.arange(h), best_fits] <= np.diagonal(conflicts[k])
                ).all():
                    print(f"{k} shape param match is out of order")
                    continue
        mappings.update(
            {old_name: new_name for old_name, new_name in zip(v["old"], v["new"])}
        )

    return mappings


def import_state_dict(old: Dict, new_: Dict) -> Dict:
    """
    Use heuristics to load weights from original paper to current weights.
    Models haven't changed much from the original, mostly ordered stayed the same.

    **The matching heuristics are as follows**:
    
    1. Match according to weight shapes (model must have the same shapes and the same number of weight for each shape)
    2. In case more than one weight share shape, perform text similarity on weight name, make sure that the order of old weights matches the order of new weights

    """
    mappings = reconcile_weights(old, new_)
    sdict = {mappings[k]: v for k, v in old.items()}
    return sdict


