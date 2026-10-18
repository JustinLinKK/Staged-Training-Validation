# Installation

Model Preflight is distributed from this Git repository and local wheel/source archives.
There is no public PyPI release. Do not use `pip install model-preflight` from PyPI: that name
belongs to an unrelated project. The import remains `model_preflight`, and the command remains
`model-preflight`.

## Clone and install

Use Python 3.10+ on Linux or WSL2. Native Windows is not supported by the POSIX worker isolation
implementation. A dedicated virtual environment avoids conflicts with unrelated packages.

```bash
git clone https://github.com/JustinLinKK/nn-model-preflight-checker.git
cd nn-model-preflight-checker
git checkout v0.2.0
python -m venv .venv
. .venv/bin/activate
python -m pip install .
model-preflight --version
```

If an appropriate environment is already active, the installation step is just
`python -m pip install .`. It installs the checker and any missing declared dependencies;
an existing compatible PyTorch installation is reused. The package does not require Conda.

## Install directly from Git

This single command installs the paper version without a managed checkout; Git must be present:

```bash
python -m pip install "git+https://github.com/JustinLinKK/nn-model-preflight-checker.git@v0.2.0"
```

Use a full commit SHA instead of the tag for an immutable dependency pin. For a requirements
file, the equivalent direct reference is:

```text
model-preflight @ git+https://github.com/JustinLinKK/nn-model-preflight-checker.git@v0.2.0
```

Examples and research scripts are available in the repository/source distribution; the runtime
wheel contains the Python package, schemas, profiles, license, and console entry point.

## CPU-only dependency installation

The default PyTorch package selected by pip can include GPU dependencies. To prepare a small
CPU-only environment, select the CPU build first, then install the checker:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

Use the Git URL above for the second command when no checkout is present. Candidate libraries
such as Transformers, tokenizers, or dataset readers must be installed separately in this same
environment. The checker does not download candidate dependencies during a check. For paper
reproduction, record exact Python/PyTorch/library versions; the package's minimum requirements
are not an assertion that every optional architecture is supported by every PyTorch release.

## Wheel or source archive

Build artifacts from a checkout:

```bash
python -m pip install build
python -m build
python -m pip install dist/model_preflight-0.2.0-py3-none-any.whl
```

The wheel also works in an offline environment when all runtime dependencies are already
installed. A plain `pip install` from the wheel resolves dependencies normally; do not use
`--no-deps` to hide missing requirements.

## Integration and overhead

```python
from model_preflight import check, submit

report = check("preflight.yaml")
submission = submit("preflight.yaml")
initial = submission.initial.to_dict()
# Continue framework work; consume the final result later.
final = submission.completion.result().to_dict()
```

These are two independent checks; use whichever API matches the integration. In a long-lived
framework, installing once and importing the package avoids a Docker launch for every trusted
candidate. Runtime stages still use fresh subprocesses, so their PyTorch imports and simulation
costs remain. This is not a claim of a newly measured end-to-end speedup.

For untrusted generated code, retain the [CPU container workflow](agent-integration.md#cpu-container-worker)
and [security controls](security.md). All existing Dockerfiles remain unchanged in this release.
