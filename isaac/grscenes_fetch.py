"""Fetch single scenes out of GRScenes' split zip archives over HTTP range requests.

home_scenes.zip is a 3-part split archive (z01, z02, zip; 51 GB). Instead of
downloading it all, read the central directory from the tail of the last part,
then fetch only the entries a scene needs: its own folder, the models and
materials its metadata.json references.

    python3 isaac/grscenes_fetch.py list                  # scenes with their referenced-asset sizes
    python3 isaac/grscenes_fetch.py get <scene> ~/grscenes  # e.g. MVUHLWYKTKJ5EAABAAAAAAI8_usd
    python3 isaac/grscenes_fetch.py meta <scene>            # metadata and file list of one scene

`get` writes <out>/home_scenes/scenes/<scene>/ plus the models and materials it references (about 1 GB),
and skips files that are already complete, so an interrupted fetch can be rerun. The central directory
is cached in ~/grscenes/.central_directory.json. GRScenes is CC BY-NC-SA 4.0 (non-commercial use only).
"""
import io
import json
import os
import stat
import struct
import sys
import zlib
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/datasets/InternRobotics/GRScenes/resolve/main/scenes/GRScenes-100/"
PARTS = ["home_scenes.z01", "home_scenes.z02", "home_scenes.zip"]
CACHE = os.path.expanduser("~/grscenes/.central_directory.json")


def http(url, start=None, end=None):
    import time
    from urllib.error import HTTPError, URLError
    for attempt in range(12):
        req = Request(url, headers={"User-Agent": "grscenes-fetch"})
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end}")
        try:
            with urlopen(req, timeout=120) as r:
                return r.read()
        except HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == 11:
                raise
            wait = float(err.headers.get("Retry-After") or min(300, 10 * 2 ** attempt))
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 11:
                raise
            wait = min(300, 10 * 2 ** attempt)
        time.sleep(wait)


def part_sizes():
    sizes = []
    for p in PARTS:
        req = Request(BASE + p, method="HEAD", headers={"User-Agent": "grscenes-fetch"})
        with urlopen(req, timeout=60) as r:
            sizes.append(int(r.headers["Content-Length"]))
    return sizes


class SplitStream:
    """Byte-addressable view over the concatenated parts (absolute offsets)."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.total = sum(sizes)

    def read(self, pos, length):
        out = io.BytesIO()
        while length > 0:
            disk = max(i for i, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[disk]
            n = min(length, self.sizes[disk] - local)
            out.write(http(BASE + PARTS[disk], local, local + n - 1))
            pos += n
            length -= n
        return out.getvalue()

    def abs_offset(self, disk, local):
        return self.starts[disk] + local


def read_central_directory(stream):
    tail = stream.read(stream.total - 65536 - 56 - 20, 65536 + 56 + 20)
    base = stream.total - len(tail)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found")
    _, disk, cd_disk, _, n_total, cd_size, cd_off, _ = struct.unpack("<4sHHHHIIH", tail[i:i + 22])
    j = tail.rfind(b"PK\x06\x07", 0, i)       # zip64 locator
    if j >= 0 and (cd_off == 0xFFFFFFFF or n_total == 0xFFFF or cd_disk == 0xFFFF):
        _, z64_disk, z64_off, _ = struct.unpack("<4sIQI", tail[j:j + 20])
        rec = stream.read(stream.abs_offset(z64_disk, z64_off), 56)
        (_, _, _, _, _, _, _, n_total, cd_size, cd_off) = struct.unpack("<4sQHHIIQQQQ", rec)
        cd_disk = struct.unpack("<I", rec[20:24])[0]
    cd = stream.read(stream.abs_offset(cd_disk, cd_off), cd_size)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, dstart, _, ext, loff) = \
            struct.unpack("<4sHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            tag, size = struct.unpack("<HH", extra[q:q + 4])
            if tag == 1:   # zip64: 8-byte usize, csize, loff and 4-byte disk, each only if its field is saturated
                vals, r = [], q + 4
                for current, sentinel, fmt in ((usize, 0xFFFFFFFF, "<Q"), (csize, 0xFFFFFFFF, "<Q"),
                                               (loff, 0xFFFFFFFF, "<Q"), (dstart, 0xFFFF, "<I")):
                    if current == sentinel:
                        width = struct.calcsize(fmt)
                        vals.append(struct.unpack(fmt, extra[r:r + width])[0]); r += width
                    else:
                        vals.append(current)
                usize, csize, loff, dstart = vals
            q += 4 + size
        entries.append(dict(name=name, method=method, csize=csize, usize=usize, crc=crc,
                            disk=dstart, loff=loff, is_link=stat.S_ISLNK(ext >> 16), is_dir=name.endswith("/")))
        p += 46 + nlen + elen + clen
    return entries


def load_entries():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            d = json.load(f)
        return SplitStream(d["sizes"]), d["entries"]
    sizes = part_sizes()
    stream = SplitStream(sizes)
    entries = read_central_directory(stream)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(dict(sizes=sizes, entries=entries), f)
    return stream, entries


def fetch_entry(stream, e):
    pos = stream.abs_offset(e["disk"], e["loff"])
    head = stream.read(pos, 30)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {e['name']}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    data = stream.read(pos + 30 + nlen + elen, e["csize"]) if e["csize"] else b""
    if e["method"] == 8:
        data = zlib.decompress(data, -15)
    elif e["method"] != 0:
        raise RuntimeError(f"unsupported compression {e['method']} for {e['name']}")
    if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
        raise RuntimeError(f"CRC mismatch for {e['name']}")
    return data


def write_entry(out, e, data):
    path = os.path.join(out, e["name"])
    if e["is_dir"]:
        os.makedirs(path, exist_ok=True); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if e["is_link"]:
        if os.path.lexists(path):
            os.remove(path)
        os.symlink(data.decode(), path)
    else:
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)


def have_entry(out, e):
    path = os.path.join(out, e["name"].rstrip("/"))
    if e["is_dir"]:
        return os.path.isdir(path)
    if e["is_link"]:
        return os.path.islink(path)
    return os.path.isfile(path) and not os.path.islink(path) and os.path.getsize(path) == e["usize"]


def main():
    cmd = sys.argv[1]
    stream, entries = load_entries()
    by_name = {e["name"]: e for e in entries}
    root = next(e["name"].split("/")[0] for e in entries if "/scenes/" in e["name"])
    scenes = sorted({e["name"].split("/")[2] for e in entries if e["name"].startswith(f"{root}/scenes/") and e["name"].count("/") >= 3})
    if cmd == "list":
        print(f"{len(entries)} entries, root {root}, {len(scenes)} scenes; parts {[s / 1e9 for s in stream.sizes]} GB")
        rows = []
        for s in scenes:
            meta = by_name.get(f"{root}/scenes/{s}/metadata.json")
            own = sum(e["usize"] for e in entries if e["name"].startswith(f"{root}/scenes/{s}/"))
            refs = json.loads(fetch_entry(stream, meta)) if meta else {}
            ref_paths = set()
            for v in refs.values() if isinstance(refs, dict) else []:
                for item in (v if isinstance(v, list) else [v]):
                    if isinstance(item, str):
                        ref_paths.add(item)
            ref_size = sum(e["usize"] for e in entries if any(e["name"].startswith(f"{root}/{p.lstrip('./').lstrip('../')}") for p in ref_paths)) if ref_paths else -1
            rows.append((own + max(ref_size, 0), own, ref_size, len(ref_paths), s))
            print(f"{s:30s} own {own/1e6:7.1f} MB  refs {ref_size/1e9:6.2f} GB ({len(ref_paths)} paths)", flush=True)
        rows.sort()
        print("smallest:", [r[4] for r in rows[:5]])
    elif cmd == "get":
        scene, out = sys.argv[2], sys.argv[3]
        meta = json.loads(fetch_entry(stream, by_name[f"{root}/scenes/{scene}/metadata.json"]))
        wanted = [e for e in entries if e["name"].startswith(f"{root}/scenes/{scene}/")]
        ref_paths = set()
        for v in meta.values() if isinstance(meta, dict) else []:
            for item in (v if isinstance(v, list) else [v]):
                if isinstance(item, str):
                    ref_paths.add(item.lstrip("./").lstrip("../"))
        # Referenced files, plus everything in each referenced model folder: the folders hold a
        # `Materials` symlink to the shared library, which the textures resolve through.
        files = {f"{root}/{p}" for p in ref_paths}
        dirs = {f"{root}/{os.path.dirname(p)}" for p in ref_paths if p.startswith("models/")}
        for e in entries:
            name = e["name"].rstrip("/")
            if name in files or os.path.dirname(name) in dirs:
                wanted.append(e)
        uniq = {e["name"]: e for e in wanted}
        have = [e for e in uniq.values() if have_entry(out, e)]
        todo = [e for e in uniq.values() if not have_entry(out, e)]
        for e in todo:                      # directories first, files in parallel
            if e["is_dir"]:
                write_entry(out, e, b"")
        files = [e for e in todo if not e["is_dir"]]
        total = sum(e["usize"] for e in files)
        print(f"{len(uniq)} entries, {len(have)} already present; {len(files)} files, {total/1e9:.2f} GB to fetch", flush=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def job(e):
            write_entry(out, e, fetch_entry(stream, e)); return e
        done, n = 0, 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for fut in as_completed([pool.submit(job, e) for e in files]):
                e = fut.result(); done += e["usize"]; n += 1
                if n % 100 == 0 or n == len(files):
                    print(f"  {n}/{len(files)} files  {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
        print("done", os.path.join(out, root, "scenes", scene))
    elif cmd == "meta":
        scene = sys.argv[2]
        print(fetch_entry(stream, by_name[f"{root}/scenes/{scene}/metadata.json"]).decode()[:3000])
        for e in entries:
            if e["name"].startswith(f"{root}/scenes/{scene}/"):
                print(f"  {e['usize']:12d} {'L' if e['is_link'] else ' '} {e['name']}")


if __name__ == "__main__":
    main()
