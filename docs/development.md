# Development and repository layout

| Path | Purpose |
|---|---|
| `src/model_preflight/` | Installable checker API, CLI, workers, schemas, and profiles |
| `tests/` | Package unit, isolation, integration, and mutation tests |
| `tests/compatibility/` | Lightning compatibility plugin tests and their pytest import setup |
| `examples/` | Candidate adapter and manifest examples |
| `scripts/` | Corpus evaluation, reduction experiments, and concurrency benchmark |
| `docker/` | Full CPU checker and optional CUDA-build images |
| `docs/` | Versioned documentation wiki, paper evidence, and historical design |
| Root `preflight.py`, `plugin.py`, `cpu_probe.py`, `Dockerfile` | Separate Lightning compatibility plugin; paths preserved for consumers |

The package and compatibility plugin have distinct interfaces. Do not move the compatibility
modules into the runtime package or change their import paths as a documentation cleanup.

## Editable setup and verification

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
ruff check .
mypy src/model_preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
python -m build
```

The existing `environment.yml` is an optional Conda alternative. Optional native Mamba and
compatibility-image tests report explicit skips when their dependencies are absent.

## Installed-package verification

The version is defined once in `src/model_preflight/version.py`; setuptools reads that value
for both wheel and source archive metadata. `CITATION.cff` and the changelog identify the
research release. Manifest, report, and worker protocol versions remain independent.

Before publishing a Git release, install the wheel in a separate environment and run outside
the repository so an editable checkout cannot conceal missing packaged resources:

```bash
python -m venv /tmp/model-preflight-wheel-test
/tmp/model-preflight-wheel-test/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/model-preflight-wheel-test/bin/python -m pip install dist/model_preflight-0.2.1-py3-none-any.whl
cd /tmp
/tmp/model-preflight-wheel-test/bin/model-preflight --version
```

Run a complete check and speculative submission against an absolute manifest path. Verify
the emitted report schema, packaged profiles, expected late failures, and entry-point version.
The source archive also includes docs, examples, tests, experiment scripts, and compatibility
sources so its own test suite and builds remain usable.

Paper-reproduction commands and interpretation limits are in [paper-release.md](paper-release.md).
Do not overwrite frozen experiment directories when rerunning a study.
