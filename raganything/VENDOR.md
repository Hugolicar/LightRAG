# Vendored RAG-Anything

- Upstream: https://github.com/HKUDS/RAG-Anything
- Version: 1.3.1
- Commit: 32eef6ecc2cc9d84befe9fa042b0559b2a862901
- Commit date: 2026-06-15
- License: MIT; see `LICENSE` in this directory.

## Why this is vendored

This fork vendors the `raganything/` Python package instead of depending on the
PyPI `raganything` distribution so Railway builds do not accidentally resolve a
conflicting LightRAG runtime. The bridge imports RAG-Anything lazily and remains
disabled by default behind `RAGANYTHING_ENABLE=false`.
