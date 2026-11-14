#!/usr/bin/env python3
"""Unpack the Nuitka "onefile" build of Marathon Lab (Linux, x86-64).

The download is a small ELF bootstrap with a compressed payload appended:

    [ ELF bootstrap ][ b"KA" b"Y" ][ zstd stream ][ u64 payload_size ]

payload_size is little-endian and counts the bytes from the b"KA" magic up to
the end of the zstd stream. b"Y" means "compressed" (b"X" would be raw).

The decompressed stream is a flat list of files:

    name (UTF-8, NUL-terminated) | flags (1 byte, bit 0 = executable) |
    size (u64, little-endian)    | data (size bytes)

and ends with an empty name. The real program is ``marathon-lab.bin``; the
rest are its shared libraries plus one data file, ``lab/proven.npy``.

Usage:
    python3 extract_onefile.py marathon-lab-linux-x86_64 OUT_DIR
"""
import os
import struct
import sys

try:
    import zstandard
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: pip install zstandard")


def read_payload(path):
    data = open(path, "rb").read()
    (payload_size,) = struct.unpack("<Q", data[-8:])
    start = len(data) - 8 - payload_size
    magic = data[start:start + 3]
    if magic[:2] != b"KA":
        raise ValueError("no Nuitka onefile payload found (magic %r)" % magic)
    body = data[start + 3:len(data) - 8]
    if magic[2:3] == b"Y":
        return zstandard.ZstdDecompressor().stream_reader(body).read()
    return body


def unpack(path, out_dir):
    raw = read_payload(path)
    out_dir = os.path.abspath(out_dir)
    entries = []
    p = 0
    while p < len(raw):
        end = raw.index(b"\0", p)
        name = raw[p:end].decode("utf-8")
        if not name:
            break
        flags = raw[end + 1]
        (size,) = struct.unpack("<Q", raw[end + 2:end + 10])
        start = end + 10
        target = os.path.abspath(os.path.join(out_dir, name))
        if not target.startswith(out_dir + os.sep):
            raise ValueError("refusing to write outside the output folder: %r" % name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(raw[start:start + size])
        if flags & 1:
            os.chmod(target, 0o755)
        entries.append((name, size))
        p = start + size
    return entries


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    entries = unpack(sys.argv[1], sys.argv[2])
    names = {n for n, _ in entries}
    assert "marathon-lab.bin" in names, "main program not found in payload"
    assert "lab/proven.npy" in names, "bundled data file not found in payload"
    total = sum(s for _, s in entries)
    print("unpacked %d files (%.1f MB) into %s" % (len(entries), total / 1e6, sys.argv[2]))


if __name__ == "__main__":
    main()
