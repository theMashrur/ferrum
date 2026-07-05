# ferrum

`ferrum` is a Rust-first computational linear algebra library with a Python interface.

The project is structured as a high-performance core crate (Rust) plus a thin Python boundary (PyO3/maturin). The repository currently focuses on architecture, packaging, and verification scaffolding, so it is intentionally incomplete while still reflecting a realistic library layout.

## Status

- Stage: pre-alpha scaffold
- Target: dense/spectral matrix routines with parallel Rust execution
- Python package: `ferrum`
- Rust crate type: `cdylib` extension module

## Design goals

- Keep heavy numerical work on Rust-owned memory
- Expose a clean Python surface without leaking Rust internals
- Make multithreaded CPU execution practical behind Python APIs
- Grow algorithms incrementally with test-first checkpoints

## Current capabilities

- Build and package pipeline for a Rust-backed Python extension
- Module boundaries for core structures, algorithms, bindings, and threading
- Import smoke test for validating Python <-> Rust wiring

## Planned algorithm track

- Arithmetic primitives
- QR decomposition
- FFT routines
- LU decomposition
- Schur decomposition
- Singular value decomposition
- Cholesky decomposition
- Least squares via QR

## Repository layout

```text
ferrum/
  Cargo.toml
  pyproject.toml
  python/ferrum/__init__.py
  src/
    lib.rs
    algorithms/mod.rs
    bindings/mod.rs
    core/mod.rs
    threading/mod.rs
  tests/python/test_import.py
```

## Quick start

Use a virtual environment and run the smallest validation loop first:

```powershell
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
cargo check
.\.venv\Scripts\maturin.exe develop
.\.venv\Scripts\pytest.exe
```

## Architecture notes

`ferrum` treats the Python/Rust boundary as orchestration code. Numerical kernels should operate over Rust-native data structures, and Python object interaction should remain localized at API edges. This supports predictable performance and safer multithreading behavior.

The CPython GIL does not block native Rust threading for CPU-bound work that stays on Rust-managed memory. That makes decomposition and transform kernels good candidates for internal parallel execution strategies.

## License

This project is dual-licensed under either:

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

You may choose either license when using this project.
