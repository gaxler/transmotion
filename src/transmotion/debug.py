"""
This is an ad-hoc code used for debugging gaps in the dense motion generation network.
Can safely ignore, its here "for my records"
"""
import torch as th
from typing import Callable, Any, Dict, Sequence
from pathlib import Path


class MMapDebug:

    _total_runs = 0

    @staticmethod
    def _map_over(
        fn: Callable[[th.Tensor], Any],
        t: th.Tensor | Dict[str, th.Tensor] | Sequence[th.Tensor],
    ):
        if isinstance(t, Dict):
            new_t = {k: fn(v) for k, v in t.items()}
        elif isinstance(t, Sequence):
            new_t = [fn(v) for v in t]
        elif isinstance(t, th.Tensor):
            new_t = fn(t)
        else:
            raise Exception

        return new_t

    def __init__(self, fpath: str, disable=False) -> None:
        self._tensors = []
        self._comp_tensors = []
        cur_run = MMapDebug._total_runs
        self._fpath = Path(fpath) / f"run_{cur_run}.pkl"
        self._comp_results = []

        self.disable = disable

    def push(self, t: th.Tensor | Dict[str, th.Tensor] | Sequence[th.Tensor]):
        if self.disable:
            return

        def _prep(t):
            return t.detach().cpu().clone()

        new_t = self._map_over(_prep, t)

        self._tensors.append(new_t)

    def comp(self, t):
        if not len(self._comp_tensors) or self.disable:
            return
        idx = len(self._tensors)

        # def _make_comp(t):

        def _comp(other_t):
            res = {
                "l1": th.abs(t.flatten() - other_t.flatten()).sum()
                / (t.abs().sum() + 1e-9),
                "mean": (t.mean().item(), other_t.mean().item()),
                "vars": (t.var().item(), other_t.var().item()),
            }
            return res

        #     return _comp

        # comps_t = self._map_over(_make_comp, tens_)

        other = self._comp_tensors[idx]
        res = _comp(other)

        self._tensors.append(t)
        self._comp_results.append(res)

        return res

    def __enter__(self):
        if self._fpath.exists() and not self.disable:
            self._comp_tensors = th.load(self._fpath)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.disable:
            return
        if len(self._comp_tensors):
            print(f"==== RUN {MMapDebug._total_runs} ===")
            for idx, comp in enumerate(self._comp_results):
                print(f"\t({idx}):\n\t {comp}")
        else:
            th.save(self._tensors, self._fpath)
            print(f"Saved Debug run: {MMapDebug._total_runs}")
        MMapDebug._total_runs += 1
        return
