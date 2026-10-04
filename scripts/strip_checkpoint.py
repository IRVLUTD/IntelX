#!/usr/bin/env python3
"""Rewrite a training checkpoint as a weights-only FP16 state dict.

    python scripts/strip_checkpoint.py weights/checkpoint_best.pth \
                                       weights/intelx_unet_resnet34_fp16.pt

An optional 49 MB copy of the released checkpoint, for inference only. It
needs numpy and the standard library. It does not need torch, and it never
executes the pickle it is reading.

What it does, and why each part is necessary.

  Drops the optimizer state.   AdamW keeps two moments per parameter, which is
                               two thirds of a training checkpoint and of no
                               use to anyone running inference.
  Casts floats to FP16.        Halves what is left. FP16 tops out at 65504, so
                               a tensor whose values run past that is left at
                               FP32 instead. In this model thirteen BatchNorm
                               running_var buffers do, the largest reaching
                               4.0e5, and casting them would turn them into
                               inf and poison every activation downstream.
  Drops the _orig_mod. prefix. torch.compile adds it to every key. A plain
                               model does not have it, so a checkpoint that
                               keeps it will not load into one.
  Moves storages to cpu.       The original recorded cuda:0, which forces
                               map_location on a machine without a GPU.
  Rebuilds the pickle.         A pickle stores every string it was given and
                               the module path of every class it references.
                               Writing a fresh one leaves nothing behind but
                               tensors, so no path or name rides along.

Check the result with scripts/inspect_checkpoint.py.

A checkpoint is a zip holding a protocol-2 pickle plus one raw blob per tensor
storage. This reads that structure with numpy and the standard library, keeps
only the model weights, casts the floating-point ones to FP16, drops the
`_orig_mod.` prefix that `torch.compile` adds, moves the storages to `cpu`, and
writes the same structure back out.

The output references only `collections.OrderedDict`, `torch.HalfStorage`,
`torch.LongStorage` and `torch._utils._rebuild_tensor_v2`, which is what a
plain state dict is made of and nothing else.
"""
import collections
import io
import os
import pickle
import sys
import zipfile

import numpy as np

DTYPE = {"FloatStorage": np.float32, "HalfStorage": np.float16,
         "DoubleStorage": np.float64, "LongStorage": np.int64,
         "IntStorage": np.int32, "ShortStorage": np.int16,
         "CharStorage": np.int8, "ByteStorage": np.uint8,
         "BoolStorage": np.bool_, "BFloat16Storage": None}


class Stub:
    def __init__(self, module, name): self.module, self.name = module, name
    def __repr__(self): return "%s.%s" % (self.module, self.name)


class StorageRef:
    def __init__(self, stype, key, location, numel):
        self.stype, self.key, self.location, self.numel = stype, key, location, numel
    def __repr__(self):
        return "Storage(%s, key=%s, numel=%d)" % (self.stype, self.key, self.numel)


class TensorRef:
    def __init__(self, storage, offset, size, stride, requires_grad=False):
        self.storage, self.offset = storage, offset
        self.size, self.stride = tuple(size), tuple(stride)
        self.requires_grad = requires_grad
    @property
    def dtype(self): return DTYPE[self.storage.stype]
    def __repr__(self):
        return "Tensor%s %s" % (list(self.size), self.storage.stype)


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, *a):
    return TensorRef(storage, offset, size, stride, requires_grad)


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2",
                                                 "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda t, rg, hooks: t
        if module.startswith("torch"):
            return Stub(module, name)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, stype, key, location, numel = pid
        assert kind == "storage", kind
        return StorageRef(stype.name if isinstance(stype, Stub) else str(stype),
                          key, location, numel)


def load(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    obj = Reader(io.BytesIO(z.read(prefix + "/data.pkl"))).load()
    return obj, z, prefix


def array(t, z, prefix):
    """The tensor's values, as a numpy array."""
    raw = z.read("%s/data/%s" % (prefix, t.storage.key))
    flat = np.frombuffer(raw, dtype=t.dtype)
    n = int(np.prod(t.size)) if t.size else 1
    return flat[t.offset:t.offset + n].reshape(t.size)

FP16_SAFE_MAX = 60000.0        # a margin under fp16's 65504

STORAGE_FOR = {
    np.dtype(np.float16): "HalfStorage",
    np.dtype(np.float32): "FloatStorage",
    np.dtype(np.float64): "DoubleStorage",
    np.dtype(np.int64): "LongStorage",
    np.dtype(np.int32): "IntStorage",
    np.dtype(np.int16): "ShortStorage",
    np.dtype(np.int8): "CharStorage",
    np.dtype(np.uint8): "ByteStorage",
    np.dtype(np.bool_): "BoolStorage",
}


class Global(object):
    """A name to emit as a GLOBAL opcode, for a module not importable here.

    save_reduce insists its func be callable, so this is, though calling it
    would be a bug."""
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *a, **k):
        raise RuntimeError("%s.%s is a name to write, not to run"
                           % (self.module, self.name))


class Out(object):
    """One tensor in the output, with the storage key it will be written to."""
    def __init__(self, key, arr):
        self.key, self.arr = key, arr


def c_strides(shape):
    s, acc = [], 1
    for d in reversed(shape):
        s.append(acc)
        acc *= int(d)
    return tuple(reversed(s))


class TorchPickler(pickle._Pickler):
    """protocol 2, GLOBAL for torch names, BINPERSID for storages."""

    def __init__(self, f):
        pickle._Pickler.__init__(self, f, protocol=2)

    def persistent_id(self, obj):
        if isinstance(obj, Out):
            return ("storage", Global("torch", STORAGE_FOR[obj.arr.dtype]),
                    obj.key, "cpu", int(obj.arr.size))
        return None

    def save(self, obj, save_persistent_id=True):
        if isinstance(obj, Global):
            self.framer.commit_frame()
            x = self.memo.get(id(obj))
            if x is not None:
                self.write(self.get(x[0]))
                return
            self.write(pickle.GLOBAL
                       + ("%s\n%s\n" % (obj.module, obj.name)).encode("utf-8"))
            self.memoize(obj)
            return
        if isinstance(obj, Out) and save_persistent_id:
            return pickle._Pickler.save(self, obj, save_persistent_id)
        return pickle._Pickler.save(self, obj, save_persistent_id)

    def save_tensor(self, t):
        shape = tuple(int(d) for d in t.arr.shape)
        self.save_reduce(Global("torch._utils", "_rebuild_tensor_v2"),
                         (t, 0, shape, c_strides(shape), False,
                          collections.OrderedDict()),
                         obj=object())


def build_pickle(entries):
    """entries: [(name, Out)] -> the bytes of data.pkl"""
    buf = io.BytesIO()
    p = TorchPickler(buf)
    p.write(pickle.PROTO + bytes([2]))

    # OrderedDict()  then  SETITEMS, exactly the shape torch writes
    od = collections.OrderedDict()
    p.save_reduce(Global("collections", "OrderedDict"), (), obj=od)
    p.write(pickle.MARK)
    for name, out in entries:
        p.save(name)
        p.save_tensor(out)
    p.write(pickle.SETITEMS)
    p.write(pickle.STOP)
    return buf.getvalue()


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: python scripts/strip_checkpoint.py "
                 "<in.pth> <out.pt>")
    src, dst = argv

    obj, z, prefix = load(src)
    sd = obj.get("model_state_dict", obj)
    print("read %s\n  %d entries, %d other top-level keys"
          % (os.path.basename(src), len(sd),
             len(obj) - 1 if "model_state_dict" in obj else 0))

    # fp16 tops out at 65504. BatchNorm running_var runs far past that, and an
    # overflow there becomes inf and poisons every activation through that
    # layer, so the decision is made on the values rather than on the name.
    entries, casts, ints, wide = [], 0, 0, []
    for i, (name, t) in enumerate(sd.items()):
        if not isinstance(t, TensorRef):
            sys.exit("%s is not a tensor, refusing to guess" % name)
        a = array(t, z, prefix)
        if a.dtype.kind != "f":
            ints += 1
        elif a.size and float(np.abs(a).max()) > FP16_SAFE_MAX:
            wide.append((name, float(np.abs(a).max()), a.size))
        else:
            a = a.astype(np.float16)
            casts += 1
        clean = name.replace("_orig_mod.", "")
        entries.append((clean, Out(str(i), np.asarray(a, order="C"))))
    print("  %d float tensors cast to fp16" % casts)
    print("  %d integer tensors kept as they were" % ints)
    print("  %d float tensors kept at fp32 because they exceed %g:"
          % (len(wide), FP16_SAFE_MAX))
    for n, m, s_ in sorted(wide, key=lambda x: -x[1]):
        print("      %-50s max %.3g" % (n.replace("_orig_mod.", ""), m))
    renamed = sum(1 for n, _ in sd.items() if "_orig_mod." in n)
    print("  dropped the _orig_mod. prefix from %d keys" % renamed)

    blob = build_pickle(entries)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as out:
        base = os.path.splitext(os.path.basename(dst))[0]
        out.writestr(base + "/data.pkl", blob)
        out.writestr(base + "/byteorder", b"little")
        for _, o in entries:
            out.writestr("%s/data/%s" % (base, o.key), o.arr.tobytes())
        out.writestr(base + "/version", b"3\n")

    print("wrote %s  %.1f MB  (was %.1f MB)"
          % (dst, os.path.getsize(dst) / 1e6, os.path.getsize(src) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
