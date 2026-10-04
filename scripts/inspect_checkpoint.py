#!/usr/bin/env python3
"""Report what a .pt or .pth checkpoint carries besides tensors.

    python scripts/inspect_checkpoint.py weights/checkpoint_best.pth

A checkpoint is a zip holding a pickle. The pickle records the module path of
every class it references and every string it stores, so a training checkpoint
routinely carries the argparse namespace it was launched with, the output
directory it wrote to, a run tracker's project and entity, and the import path
of the module the model class was defined in. None of that is tensor data and
all of it can name a person, a lab or a machine.

This reads the pickle with `pickletools` rather than `torch.load`, so it needs
nothing installed and it never executes what it is inspecting. Run it on a
checkpoint before you commit it, and again after you strip it.
"""
import os
import pickletools
import re
import sys
import zipfile

LEAKS = [
    ("a home directory", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("a Windows user profile", re.compile(r"[A-Za-z]:\\Users\\")),
    ("an email address",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("a code host URL",
     re.compile(r"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.com/[\w.-]+")),
    ("a run tracker", re.compile(r"\b(?:wandb|comet_ml|neptune|mlflow)", re.I)),
    ("a routable host",
     re.compile(r"(?:[\"'@]|//)(?!127\.0\.0\.1\b|0\.0\.0\.0\b)"
                r"(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# Everything a pure state dict is allowed to reach for.
BENIGN_CLASSES = {
    "collections.OrderedDict",
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter",
    "torch.FloatStorage", "torch.HalfStorage", "torch.DoubleStorage",
    "torch.LongStorage", "torch.IntStorage", "torch.BoolStorage",
    "torch.BFloat16Storage", "torch.ByteStorage",
    "torch.storage._load_from_bytes",
}


def pickles_in(path):
    """Yield (name, bytes) for every pickle in the file."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            yield "__zip_entry_names__", "\n".join(names).encode()
            for n in names:
                if n.endswith(".pkl") or n.rsplit("/", 1)[-1] == "data.pkl":
                    yield n, z.read(n)
    else:                                  # pre-1.6 flat pickle
        yield os.path.basename(path), open(path, "rb").read()


def walk(blob):
    """Every class reference and every string literal in one pickle."""
    classes, strings = set(), []
    try:
        for op, arg, _ in pickletools.genops(blob):
            if op.name in ("GLOBAL", "STACK_GLOBAL") and arg:
                classes.add(str(arg).replace(" ", "."))
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE",
                             "SHORT_BINBYTES", "BINBYTES", "STRING",
                             "BINSTRING", "SHORT_BINSTRING") and arg:
                s = arg.decode("utf-8", "replace") if isinstance(arg, bytes) else arg
                if len(s) > 3:
                    strings.append(s)
    except Exception as e:                 # a truncated tensor blob, not a pickle
        pass
    return classes, strings


def main(argv):
    if len(argv) != 1:
        sys.exit("usage: python scripts/inspect_checkpoint.py <checkpoint>")
    path = argv[0]
    if not os.path.exists(path):
        sys.exit("no such file: %s" % path)

    print("%s  %.1f MB\n" % (path, os.path.getsize(path) / 1e6))

    all_classes, findings, n_strings = set(), [], 0
    for name, blob in pickles_in(path):
        if name == "__zip_entry_names__":
            text = blob.decode("utf-8", "replace")
            prefix = text.split("\n")[0].split("/")[0]
            print("archive prefix  %s\n    torch takes this from the file name it "
                  "was saved under, so it\n    shows what the checkpoint was "
                  "originally called.\n" % prefix)
            for what, rx in LEAKS:
                m = rx.search(text)
                if m:
                    findings.append(("zip entry names", what, m.group(0)))
            continue
        classes, strings = walk(blob)
        all_classes |= classes
        n_strings += len(strings)
        for s in strings:
            for what, rx in LEAKS:
                m = rx.search(s)
                if m:
                    findings.append((name, what, m.group(0)))

    foreign = sorted(all_classes - BENIGN_CLASSES)
    print("classes referenced by the pickle")
    if not all_classes:
        print("    none, this is raw tensor data")
    for c in sorted(all_classes):
        print("    %-46s %s" % (c, "" if c in BENIGN_CLASSES else "<-- not a plain tensor"))
    if foreign:
        print("\n    %d of these are not part of a plain state dict. A class from"
              "\n    your own tree records the module path it was defined in, and a"
              "\n    config or argparse object stores whatever it was launched with."
              % len(foreign))

    print("\n%d string literals scanned" % n_strings)
    uniq, seen = [], set()
    for where, what, hit in findings:
        if (what, hit) in seen:
            continue
        seen.add((what, hit))
        uniq.append((where, what, hit))
    if uniq:
        print("\nFOUND %d thing(s) that should not ship" % len(uniq))
        for where, what, hit in uniq:
            print("    %-26s %-24s %s" % (where.split("/")[0], what, hit))
    else:
        print("no home directories, emails, hosts or tracker names")

    clean = not findings and not foreign
    print("\n%s" % ("CLEAN. Nothing here but tensors." if clean else
                    "NOT CLEAN. Re-save weights only, see weights/README.md."))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
