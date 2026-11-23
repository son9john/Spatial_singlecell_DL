"""Minimal read-only HDF5 access through the system libhdf5 (ctypes).

Used only because h5py is not installable in this offline workspace. Supports
numeric datasets and fixed- or variable-length string datasets, which is all a
10x Genomics feature-barcode matrix or a simple exported table needs.
"""
import ctypes
import ctypes.util
import numpy as np

_LIB_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.103",
    ctypes.util.find_library("hdf5_serial") or "",
    ctypes.util.find_library("hdf5") or "",
)

for _path in _LIB_CANDIDATES:
    if _path:
        try:
            _lib = ctypes.CDLL(_path)
            break
        except OSError:
            continue
else:  # pragma: no cover
    raise ImportError("libhdf5 not found")

hid_t = ctypes.c_int64
herr_t = ctypes.c_int
hsize_t = ctypes.c_ulonglong
H5P_DEFAULT = hid_t(0)
H5S_ALL = hid_t(0)
H5F_ACC_RDONLY = ctypes.c_uint(0)
H5T_INTEGER, H5T_FLOAT, H5T_STRING = 0, 1, 3

_lib.H5open.restype = herr_t
_lib.H5open()
_lib.H5Eset_auto2.argtypes = [hid_t, ctypes.c_void_p, ctypes.c_void_p]
_lib.H5Eset_auto2(hid_t(0), None, None)  # silence the C error stack printer

for name, res, args in [
    ("H5Fopen", hid_t, [ctypes.c_char_p, ctypes.c_uint, hid_t]),
    ("H5Fclose", herr_t, [hid_t]),
    ("H5Dopen2", hid_t, [hid_t, ctypes.c_char_p, hid_t]),
    ("H5Dclose", herr_t, [hid_t]),
    ("H5Dget_space", hid_t, [hid_t]),
    ("H5Dget_type", hid_t, [hid_t]),
    ("H5Dread", herr_t, [hid_t, hid_t, hid_t, hid_t, hid_t, ctypes.c_void_p]),
    ("H5Dvlen_reclaim", herr_t, [hid_t, hid_t, hid_t, ctypes.c_void_p]),
    ("H5Sget_simple_extent_ndims", ctypes.c_int, [hid_t]),
    ("H5Sget_simple_extent_dims", ctypes.c_int, [hid_t, ctypes.POINTER(hsize_t), ctypes.POINTER(hsize_t)]),
    ("H5Sclose", herr_t, [hid_t]),
    ("H5Tget_class", ctypes.c_int, [hid_t]),
    ("H5Tget_size", ctypes.c_size_t, [hid_t]),
    ("H5Tget_sign", ctypes.c_int, [hid_t]),
    ("H5Tis_variable_str", ctypes.c_int, [hid_t]),
    ("H5Tcopy", hid_t, [hid_t]),
    ("H5Tset_size", herr_t, [hid_t, ctypes.c_size_t]),
    ("H5Tclose", herr_t, [hid_t]),
    ("H5Lexists", ctypes.c_int, [hid_t, ctypes.c_char_p, hid_t]),
]:
    fn = getattr(_lib, name)
    fn.restype = res
    fn.argtypes = args


def _native(name):
    return hid_t.in_dll(_lib, name)


_NATIVE = {
    (H5T_INTEGER, 1, 1): ("H5T_NATIVE_SCHAR_g", np.int8),
    (H5T_INTEGER, 1, 0): ("H5T_NATIVE_UCHAR_g", np.uint8),
    (H5T_INTEGER, 2, 1): ("H5T_NATIVE_SHORT_g", np.int16),
    (H5T_INTEGER, 2, 0): ("H5T_NATIVE_USHORT_g", np.uint16),
    (H5T_INTEGER, 4, 1): ("H5T_NATIVE_INT_g", np.int32),
    (H5T_INTEGER, 4, 0): ("H5T_NATIVE_UINT_g", np.uint32),
    (H5T_INTEGER, 8, 1): ("H5T_NATIVE_LLONG_g", np.int64),
    (H5T_INTEGER, 8, 0): ("H5T_NATIVE_ULLONG_g", np.uint64),
    (H5T_FLOAT, 4, -1): ("H5T_NATIVE_FLOAT_g", np.float32),
    (H5T_FLOAT, 8, -1): ("H5T_NATIVE_DOUBLE_g", np.float64),
}


class File:
    """Context manager: ``with File(path) as f: arr = f['matrix/data']``."""

    def __init__(self, path):
        self.path = str(path)
        self.fid = _lib.H5Fopen(self.path.encode(), H5F_ACC_RDONLY, H5P_DEFAULT)
        if self.fid < 0:
            raise OSError(f"cannot open HDF5 file {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.fid is not None and self.fid >= 0:
            _lib.H5Fclose(self.fid)
            self.fid = None

    def __contains__(self, name):
        # check each path component so missing parents do not raise
        parts = name.strip("/").split("/")
        for k in range(1, len(parts) + 1):
            if _lib.H5Lexists(self.fid, "/".join(parts[:k]).encode(), H5P_DEFAULT) <= 0:
                return False
        return True

    def __getitem__(self, name):
        did = _lib.H5Dopen2(self.fid, name.encode(), H5P_DEFAULT)
        if did < 0:
            raise KeyError(name)
        try:
            sid = _lib.H5Dget_space(did)
            ndim = _lib.H5Sget_simple_extent_ndims(sid)
            dims = (hsize_t * max(ndim, 1))()
            _lib.H5Sget_simple_extent_dims(sid, dims, None)
            shape = tuple(int(dims[k]) for k in range(ndim))
            count = int(np.prod(shape)) if shape else 1
            tid = _lib.H5Dget_type(did)
            cls = _lib.H5Tget_class(tid)
            size = _lib.H5Tget_size(tid)
            try:
                if cls == H5T_STRING:
                    if _lib.H5Tis_variable_str(tid) > 0:
                        mem = _lib.H5Tcopy(_native("H5T_C_S1_g"))
                        _lib.H5Tset_size(mem, ctypes.c_size_t(-1).value)
                        buf = (ctypes.c_char_p * count)()
                        if _lib.H5Dread(did, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0:
                            raise OSError(f"read failed: {name}")
                        out = np.array([(b or b"").decode("utf-8", "replace") for b in buf], dtype=object)
                        _lib.H5Dvlen_reclaim(mem, sid, H5P_DEFAULT, buf)
                        _lib.H5Tclose(mem)
                    else:
                        buf = ctypes.create_string_buffer(count * size)
                        if _lib.H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0:
                            raise OSError(f"read failed: {name}")
                        raw = np.frombuffer(buf.raw, dtype=f"S{size}", count=count)
                        out = np.array([s.decode("utf-8", "replace") for s in raw], dtype=object)
                    return out.reshape(shape) if shape else out[0]
                sign = _lib.H5Tget_sign(tid) if cls == H5T_INTEGER else -1
                key = (cls, size, sign)
                if key not in _NATIVE:
                    raise TypeError(f"unsupported datatype {key} for {name}")
                native_name, dtype = _NATIVE[key]
                out = np.empty(shape if shape else (1,), dtype=dtype)
                if _lib.H5Dread(did, _native(native_name), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                out.ctypes.data_as(ctypes.c_void_p)) < 0:
                    raise OSError(f"read failed: {name}")
                return out if shape else out[0]
            finally:
                _lib.H5Tclose(tid)
                _lib.H5Sclose(sid)
        finally:
            _lib.H5Dclose(did)


def read_10x_h5(path, gene_expression_only=True):
    """Return (counts CSR spots x genes, barcodes, gene symbols, gene ids)."""
    from scipy import sparse
    with File(path) as f:
        shape = tuple(int(v) for v in f["matrix/shape"])
        data = f["matrix/data"]
        indices = f["matrix/indices"]
        indptr = f["matrix/indptr"]
        barcodes = f["matrix/barcodes"]
        names = f["matrix/features/name"]
        ids = f["matrix/features/id"]
        ftype = f["matrix/features/feature_type"] if "matrix/features/feature_type" in f else None
    mat = sparse.csc_matrix((data, indices, indptr), shape=shape)  # genes x barcodes
    keep = np.ones(shape[0], bool) if (ftype is None or not gene_expression_only) else (ftype == "Gene Expression")
    counts = mat[keep].T.tocsr()
    return counts, np.asarray(barcodes), np.asarray(names[keep]), np.asarray(ids[keep])
