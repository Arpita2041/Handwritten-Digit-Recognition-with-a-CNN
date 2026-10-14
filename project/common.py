"""Small utilities for auditable runs and local reports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import platform
import zipfile
from pathlib import PurePosixPath
from urllib.request import Request, urlopen
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default,
                               allow_nan=False) + "\n", encoding="utf-8")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def provenance(mode, seed, source=None):
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn",
                 "matplotlib", "joblib", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"mode": mode, "seed": seed, "utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "dependencies": versions,
            "data_sha256": sha256(source) if source else None}

def output_dirs(output):
    output = Path(output).resolve()
    for name in ("models", "reports", "audit"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output

def download_member(url, destination, member, inner_archive=None):
    """Read only the named ZIP member; never extract arbitrary archive paths."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / (destination.name + ".download")
    try:
        with urlopen(Request(url, headers={"User-Agent": "CourseProject/1.0"}),
                     timeout=60) as response, archive.open("wb") as stream:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 25 * 1024 * 1024:
                    raise ValueError("Dataset archive exceeds the expected size limit.")
                stream.write(block)
        def read_named(archive_file, name):
            if name in archive_file.namelist():
                return archive_file.read(name)
            matches = [entry for entry in archive_file.namelist()
                       if PurePosixPath(entry).name == PurePosixPath(name).name
                       and not entry.endswith("/")]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one ZIP member named {name}; found {len(matches)}.")
            return archive_file.read(matches[0])
        with zipfile.ZipFile(archive) as outer:
            if inner_archive:
                import io
                with zipfile.ZipFile(io.BytesIO(read_named(outer, inner_archive))) as inner:
                    content = read_named(inner, member)
            else:
                content = read_named(outer, member)
        staged = destination.with_suffix(destination.suffix + ".tmp")
        staged.write_bytes(content)
        staged.replace(destination)
        save_json(destination.with_suffix(destination.suffix + ".source.json"),
                  {"url": url, "member": member, "inner_archive": inner_archive,
                   "sha256": sha256(destination), "bytes": len(content)})
    except Exception as exc:
        raise RuntimeError(
            "Could not download the public dataset. Check internet access, or "
            f"download it from {url} and put {member} at {destination}. "
            f"Cause: {exc}"
        ) from exc
    finally:
        archive.unlink(missing_ok=True)
    return destination

def dataset_label(source, official_name, supplied=False):
    if Path(source).resolve().parent == (ROOT / "data/demo").resolve():
        return "SIMULATED demonstration fixture; not a real-data benchmark"
    return "User-supplied data; source not verified" if supplied else official_name

def html_report(path, title, summary, tables, images=()):
    parts = [f"<h1>{html.escape(title)}</h1><p>{html.escape(summary)}</p>"]
    for label, table in tables:
        parts.append(f"<h2>{html.escape(label)}</h2>" +
                     table.to_html(index=False, float_format=lambda x: f"{x:.4f}",
                                   escape=True, border=0))
    for filename, caption in images:
        parts.append(f'<figure><img src="{html.escape(filename, quote=True)}" '
                     f'alt="{html.escape(caption, quote=True)}"><figcaption>'
                     f'{html.escape(caption)}</figcaption></figure>')
    style = ("body{font:16px system-ui;margin:35px auto;max-width:1100px;padding:0 20px;"
             "color:#182536;background:#f7f9fc}h1,h2{color:#193b62}"
             "table{border-collapse:collapse;display:block;overflow:auto;background:white}"
             "th,td{padding:9px;border-bottom:1px solid #d8e1eb;text-align:left}"
             "img{max-width:100%;background:white}figure{margin:25px 0}"
             "p{line-height:1.6}")
    Path(path).write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
                         "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                         f"<title>{html.escape(title)}</title><style>{style}</style>"
                         + "".join(parts) + "</html>", encoding="utf-8")
