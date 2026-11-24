"""Minimal ctypes reader for numeric / fixed-string HDF5 datasets (own code; h5py is not installed)."""
import ctypes as C, numpy as np
L = C.CDLL("libhdf5_serial.so.103")
hid = C.c_int64
for f, r, a in [("H5open", C.c_int, []), ("H5Fopen", hid, [C.c_char_p, C.c_uint, hid]),
                ("H5Dopen2", hid, [hid, C.c_char_p, hid]), ("H5Dget_space", hid, [hid]),
                ("H5Sget_simple_extent_npoints", C.c_int64, [hid]), ("H5Dget_type", hid, [hid]),
                ("H5Tget_size", C.c_size_t, [hid]), ("H5Tget_class", C.c_int, [hid]),
                ("H5Tget_sign", C.c_int, [hid]), ("H5Tget_native_type", hid, [hid, C.c_int]),
                ("H5Tis_variable_str", C.c_int, [hid]),
                ("H5Dread", C.c_int, [hid, hid, hid, hid, hid, C.c_void_p]),
                ("H5Dclose", C.c_int, [hid]), ("H5Fclose", C.c_int, [hid])]:
    getattr(L, f).restype, getattr(L, f).argtypes = r, a
L.H5open()
def read(path, names):
    f = L.H5Fopen(str(path).encode(), 0, 0); assert f >= 0, path
    out = {}
    for name in names:
        d = L.H5Dopen2(f, name.encode(), 0); assert d >= 0, name
        n = L.H5Sget_simple_extent_npoints(L.H5Dget_space(d))
        t = L.H5Dget_type(d); cls, size = L.H5Tget_class(t), L.H5Tget_size(t)
        if cls == 3:
            assert L.H5Tis_variable_str(t) == 0, "variable-length strings not supported"
            buf = np.empty(n, dtype=f"S{size}"); mem = t
        else:
            mem = L.H5Tget_native_type(t, 1); size = L.H5Tget_size(mem)
            kind = "f" if cls == 1 else ("i" if L.H5Tget_sign(mem) == 1 else "u")
            buf = np.empty(n, dtype=f"{kind}{size}")
        assert L.H5Dread(d, mem, 0, 0, 0, buf.ctypes.data_as(C.c_void_p)) >= 0, name
        out[name] = buf; L.H5Dclose(d)
    L.H5Fclose(f)
    return out
