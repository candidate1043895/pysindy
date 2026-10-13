"""Make WeakSINDy subdomains avoid masked grid points."""
from contextlib import contextmanager

import numpy as np

import pysindy._weak as _weak


@contextmanager
def avoid_mask(masks, oversample=10):
    """Within this context, WeakSINDy.fit only keeps subdomains containing no
    point where mask is True.

    masks : bool array shaped like the spatiotemporal grid (st_grid.shape[:-1]),
        or a list of them (one per trajectory, same order as passed to fit).
    oversample : candidates drawn per requested subdomain before filtering.
    """
    if isinstance(masks, np.ndarray):
        masks = [masks]
    masks = iter([np.asarray(m, dtype=bool) for m in masks])
    original = _weak._make_domains

    def masked_make_domains(st_grid, subdomain_dims, n_subdomains, *args, **kwargs):
        mask = next(masks)
        if mask.shape != st_grid.shape[:-1]:
            raise ValueError(
                f"mask shape {mask.shape} != grid shape {st_grid.shape[:-1]}"
            )
        cand = original(
            st_grid, subdomain_dims, oversample * n_subdomains, *args, **kwargs
        )
        keep = [
            k
            for k, inds in enumerate(cand.axis_inds_per_subdom)
            if not mask[np.ix_(*inds)].any()
        ][:n_subdomains]
        if len(keep) < n_subdomains:
            raise RuntimeError(
                f"Only {len(keep)}/{n_subdomains} mask-free subdomains found; "
                "increase oversample, shrink H_xt, or lower n_subdomains."
            )
        return _weak.SubdomainSpecs(
            cand.domain,
            [cand.axis_inds_per_subdom[k] for k in keep],
            cand.subgrid_dims[keep],
            cand.subgrid_shapes[keep],
            [cand.subgrids_scaled[k] for k in keep],
        )

    _weak._make_domains = masked_make_domains
    try:
        yield
    finally:
        _weak._make_domains = original
