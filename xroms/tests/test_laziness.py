"""Laziness and chunk structure after every operation.

For dask-chunked input (``conftest.chunked`` splits every dim, horizontal ones too):

(a) building the result computes nothing. A dask scheduler that raises on any compute
    (like xarray's ``raise_if_dask_computes``) is installed around each call, over every
    public pure function (``_sweep.OPS``) and every ``ds.xroms`` member (``_sweep.ACCESSOR``);
(b) the result is lazy (``.chunks is not None``);
(c) chunks along dims the operation does not move equal the input's, and a dim it moves
    (rho -> u, rho -> psi, s_rho -> s_w) keeps the input's chunk boundaries, only its last
    chunk one point longer or shorter.

Exempt from (c), and why: the vertical dim of the functions that take only the grid
(``z``, ``dz``, ``dV``). They build depths from the vertical parameters (``Cs_r``, ``Cs_w``),
which are variables of their own that ``chunked`` splits differently from the data's
levels, so their levels follow the parameters' chunks (a single chunk when the parameters
are attributes, as in UCLA output). The new dim of ``zslice``/``isoslice`` has no input to
compare with. Nothing is exempt from (a) and (b).

Reading the vertical parameters is the one legitimate compute: when a parameter such as
``hc`` is itself a dask scalar (it is, for instance, in zarr stores), ``vertical_params``
reads it. ``test_dask_parameters_are_read_and_nothing_else_is`` pins that every such read is
a single number.
"""

import contextlib

import dask
import numpy as np
import pytest
import xarray as xr

import xroms
from xroms.conventions import canonicalize
from xroms.tests import _sweep as S
from xroms.tests.conftest import INPUT, chunked


class Computes:
    """A dask scheduler that refuses to compute, and counts the attempts."""

    def __init__(self):
        self.count = 0

    def __call__(self, dsk, keys, **kwargs):
        self.count += 1
        raise RuntimeError("dask data was computed while the result was being built")


@contextlib.contextmanager
def no_computes():
    counter = Computes()
    with dask.config.set(scheduler=counter):
        yield counter
    assert counter.count == 0, f"{counter.count} dask computes"


def test_the_scheduler_notices_a_compute(rutgers):
    with pytest.raises(RuntimeError, match="computed"):
        with no_computes():
            _ = chunked(rutgers).temp.values
    # ... and is gone afterwards
    assert chunked(rutgers).temp.mean().values == pytest.approx(float(rutgers.temp.mean()))


AXES = [("eta_rho", "eta_v"), ("xi_rho", "xi_u"), ("s_rho", "s_w")]


def _same_axis(dim, ref):
    for pair in AXES:
        if dim in pair:
            return next((d for d in pair if d in ref.dims), None)
    return dim if dim in ref.dims else None


def _expected(chunks, n_in, n_out):
    """``chunks`` of a dim of ``n_in`` points moved to ``n_out`` points: same boundaries, last chunk adjusted."""
    last = chunks[-1] + n_out - n_in
    return tuple(chunks[:-1]) + ((last,) if last else ())


def check_lazy(name, out, ref, *, skip_vertical=False):
    """(b) and (c) for every result in ``out``; ``ref`` is a chunked rho-point variable of the input."""
    for o in S.results(out):
        assert o.chunks is not None, f"{name}: result is not lazy"
        for dim in o.dims:
            same = _same_axis(dim, ref)
            if same is None or (skip_vertical and dim in ("s_rho", "s_w")):
                continue
            want = _expected(ref.chunksizes[same], ref.sizes[same], o.sizes[dim])
            assert o.chunksizes[dim] == want, f"{name}: chunks along {dim} are {o.chunksizes[dim]}, the input's {same} gives {want}"


@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_pure_functions_compute_nothing_and_keep_the_chunks(layout, op):
    ds, p = S.make_variant(S.dataset(layout), "chunked")
    with no_computes():
        out = op.call(ds, p)
    check_lazy(op.name, out, canonicalize(ds)["temp"], skip_vertical=op.name in S.GRID_ONLY)
    # the dtype the lazy result announces is the one it computes to, and the one numpy input gives
    if layout in ("rutgers", "ucla"):  # the vertical parameters come from variables, or from attributes
        eager = op.call(*S.make_variant(S.dataset(layout), "full"))
        for lazy, plain in zip(S.results(out), S.results(eager)):
            computed = lazy.compute()
            assert lazy.dtype == computed.dtype == plain.dtype, f"{op.name}: {lazy.dtype}, {computed.dtype}, {plain.dtype}"


#: grid-only members, whose levels follow the vertical parameters' chunks (see the module docstring)
ACCESSOR_GRID_ONLY = {"z_rho", "z_w", "z(u, s_w)", "dz", "dz(v, s_w)", "dx", "dy(u)", "dA(psi)", "dV"}


@pytest.mark.parametrize("acc", S.ACCESSOR, ids=S.ACC_IDS)
def test_accessor_computes_nothing_and_keeps_the_chunks(layout, acc):
    ds = chunked(S.dataset(layout))
    with no_computes():
        out = acc.accessor(ds)
    check_lazy(acc.name, out, canonicalize(ds)["temp"], skip_vertical=acc.name in ACCESSOR_GRID_ONLY)


def test_accessor_helpers_compute_nothing(layout):
    ds = chunked(S.dataset(layout))
    with no_computes():
        withz = ds.xroms.assign_z()
        grid = ds.xroms.xgcm_grid(vertical_metrics=True)
        sub = ds.xroms.subset(X=slice(2, 9), Y=slice(1, 7))
        params = ds.xroms.vertical_params
    assert withz["z_rho"].chunks is not None and withz["z_w"].chunks is not None
    assert set(grid.axes) == {"X", "Y", "Z"}
    assert sub["temp"].chunks is not None and sub["temp"].sizes["xi_rho"] == 7
    assert params.Cs_r.sizes["s_rho"] == ds.sizes["s_rho"]


# --- dask scalars: the parameters are read, nothing else is ------------------------------------------------


class Reads:
    """Computes as dask would, recording how many numbers each compute returns."""

    def __init__(self):
        self.sizes = []

    def __call__(self, dsk, keys, **kwargs):
        out = dask.get(dsk, keys, **kwargs)
        self.sizes.append(max(self._sizes(out)))
        return out

    def _sizes(self, x):
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from self._sizes(y)
        else:
            yield np.size(x)


def _with_dask_scalars(ds):
    """The Dataset with its 0-d parameters as dask scalars, which is how zarr stores open."""
    import dask.array as da

    names = [n for n in ("hc", "theta_s", "theta_b", "Vtransform", "Vstretching") if n in ds.variables and ds[n].ndim == 0]
    return ds.assign({n: ds[n].copy(data=da.from_array(np.asarray(ds[n].values), chunks=())) for n in names}), names


@pytest.mark.parametrize("layout", ["rutgers", "croco"])  # parameters as variables, and with VertCoordType too
@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_dask_parameters_are_read_and_nothing_else_is(layout, op):
    base = S.dataset(layout)
    ds, names = _with_dask_scalars(chunked(base))
    if not names:
        pytest.skip("this layout keeps its parameters in attributes, which are never dask arrays")
    p = lambda n: ds[n]  # noqa: E731
    reads = Reads()
    with dask.config.set(scheduler=reads):
        out = op.call(ds, p)
    assert all(o.chunks is not None for o in S.results(out)), f"{op.name}: not lazy"
    assert all(size == 1 for size in reads.sizes), f"{op.name}: computed arrays of {sorted(set(reads.sizes))} numbers; only scalar parameters may be read"
    # ... and the answer is the one the numpy-backed Dataset gives
    for got, want in zip(S.results(out), S.results(op.call(base, lambda n: base[n]))):
        np.testing.assert_allclose(got.transpose(*want.dims).values, want.values, rtol=1e-9, atol=1e-12, equal_nan=True)


# --- files opened lazily ---------------------------------------------------------------------------------


def _open_ucla():
    rst = xr.open_dataset(INPUT / "ucla_rst.nc", chunks={})
    grd = xr.open_dataset(INPUT / "ucla_grd.nc", chunks={}).drop_vars("spherical")
    return xr.merge([rst, grd], compat="override")


def _open_rutgers():
    files = [INPUT / "ocean_his_0001.nc", INPUT / "ocean_his_0002.nc"]
    return xr.open_mfdataset(files, data_vars="minimal", coords="minimal", compat="override")


@pytest.mark.parametrize("opener", [_open_ucla, _open_rutgers], ids=["ucla_rst", "rutgers_his"])
@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_files_opened_with_chunks_compute_nothing(opener, op):
    ds = opener()
    with ds:
        with no_computes():
            out = op.call(ds, lambda n: ds[n])
        assert all(o.chunks is not None for o in S.results(out)), f"{op.name}: not lazy"
        # and the result can then be computed
        assert all(np.isfinite(o.values).any() for o in S.results(out)), f"{op.name}: nothing finite"


def test_zslice_and_isoslice_rechunk_only_the_vertical(layout):
    # the vertical dim is the one an interpolation to depths has to see whole: nothing else may be touched
    ds = chunked(S.dataset(layout))
    ref = canonicalize(ds)["temp"]
    for out in (xroms.zslice(ds.temp, [-5.0, -2.0], ds), xroms.isoslice(ds.temp, [-5.0], xroms.z(ds), new_dim="zz")):
        for dim in out.dims:
            if dim in ("z", "zz"):
                assert len(out.chunksizes[dim]) == 1
            else:
                assert out.chunksizes[dim] == _expected(ref.chunksizes[dim], ref.sizes[dim], out.sizes[dim])
