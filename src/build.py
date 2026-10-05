"""Rebuild index.html by inlining src/engine.js into src/template.html.

Run from the repository root:  python src/build.py
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
template = (root / "src" / "template.html").read_text(encoding="utf-8")
engine = (root / "src" / "engine.js").read_text(encoding="utf-8")

if "/*__ENGINE__*/" not in template:
    raise SystemExit("template.html is missing the /*__ENGINE__*/ placeholder")

(root / "index.html").write_text(template.replace("/*__ENGINE__*/", engine), encoding="utf-8")
print("Wrote index.html")
