"""Extract RoBERTa-base token embeddings from spaCy's en_core_web_trf wheel, without torch.

The wheel is a GitHub release asset (explosion/spacy-models, en_core_web_trf-3.8.0).
Its transformer/model file is msgpack({"state": <torch.save bytes>, ...}); the torch.save
payload is a zip containing data.pkl plus raw tensor storages.  We read it with a
*whitelist* unpickler: only the handful of torch rebuild functions and storage types are
resolvable, and they are mapped to numpy, so no code from the file is ever executed.

Usage:  python -I scripts/extract_roberta_embeddings.py WHEEL OUT.npz
"""
import io, pickle, sys, zipfile
import numpy as np
import msgpack

DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
          "LongStorage": np.int64, "IntStorage": np.int32, "BFloat16Storage": None}


class _Storage:
    def __init__(self, kind):
        self.kind = kind


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    arr, = storage
    n = int(np.prod(size)) if len(size) else 1
    flat = arr[offset: offset + max(n, 1) * 1]  # contiguous assumption checked below
    expected = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
    if tuple(stride) != expected:
        raise ValueError(f"non-contiguous tensor {size} {stride}")
    return flat[:n].reshape(size)


class SafeUnpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch" and name in DTYPES:
            return _Storage(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _, numel = pid
        dt = DTYPES[stype.kind]
        if dt is None:
            raise ValueError("bfloat16 not supported")
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return (np.frombuffer(raw, dtype=dt)[:numel],)


def main(wheel, out):
    with zipfile.ZipFile(wheel) as w:
        name = [n for n in w.namelist() if n.endswith("transformer/model")][0]
        blob = w.read(name)
    obj = msgpack.unpackb(blob, raw=False, strict_map_key=False)
    # thinc model: obj["shims"][i] holds a PyTorchShim blob = msgpack({"config", "state"}),
    # where "state" is the torch.save zip payload
    payload = None
    for shim_list in obj["shims"]:
        for s in shim_list:
            inner = msgpack.unpackb(s, raw=False, strict_map_key=False)
            if isinstance(inner, dict) and bytes(inner.get("state", b""))[:2] == b"PK":
                payload = bytes(inner["state"])
    if payload is None:
        raise ValueError("no torch payload found")
    zf = zipfile.ZipFile(io.BytesIO(payload))
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("/data.pkl")]
    state = SafeUnpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    keys = list(state.keys())
    print(len(keys), "tensors; embedding-like keys:")
    emb = {}
    for k in keys:
        if "embed" in k.lower():
            print("  ", k, state[k].shape)
            emb[k.replace(".", "__")] = np.array(state[k], dtype=np.float32)
    np.savez_compressed(out, **emb)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
