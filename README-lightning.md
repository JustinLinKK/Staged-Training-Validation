# Lightning preflight compatibility plugin

This folder is also a standalone, versioned preflight repository. Its small
`Dockerfile` packages the existing `preflight.py` static checker behind a
one-request/one-response JSON protocol. Build with this directory as the
context, **not** the parent repository (which contains large datasets):

```bash
docker build -f Dockerfile -t lightning-preflight:local .
```

The container reads a version-1 JSON request on standard input and prints one
JSON report. Candidate Python source is parsed, never executed. The container
runs as a non-root user; `DockerStaticChecker` uses no network, GPU, writable
root filesystem, or extra Linux capabilities. A caller may mount its workspace
read-only for bounded cache-metadata checks.

The portable Python API is `check_request(request)`, `DockerStaticChecker`, and
`SpeculativePreflight.run(...)`. A framework supplies a scheduler `submit`
callback and `cancel(job_id)` callback; the checker does not assume any agent
class hierarchy or scheduler implementation.

For the Lightning ML-Master and MLEvolve adapters, set `LIGHTNING_PLUGINS` to
include `preflight,scheduler`, set `LIGHTNING_PREFLIGHT_DOCKER_IMAGE` to the
built image, and opt in with `LIGHTNING_PREFLIGHT_SPECULATIVE=1`. Supply a
trusted, JSON-encoded argument vector in
`LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV`, containing `{script_path}` and
`{epochs}` placeholders. `{working_directory}` and `{node_id}` are optional.
The adapter launches this command with `CUDA_VISIBLE_DEVICES=-1`,
`NVIDIA_VISIBLE_DEVICES=void`, and `LIGHTNING_CPU_EPOCHS=2`; no shell is used.
The command must run the actual candidate training path on a representative
CPU fixture and print a final JSON line:

```json
{"status":"passed","epochs_completed":2,"losses":[1.3,1.1]}
```

The adapter, not this generic plugin, is responsible for enforcing that two
full epochs actually ran and for isolating untrusted candidate code in its
own CPU container or restricted user. A bare success assertion is not proof
of training correctness. An explicit `{"status":"failed", "bug_confirmed":true}`
is a confirmed candidate defect and cancels its provisional scheduler job.
Timeout, missing evidence, or infrastructure failure is **inconclusive** and
does not blame the candidate or cancel the job. The scheduler may begin or
even finish a speculative job before the probe verdict; no system can undo
already executed work. A canceled queued job is excluded from the executed
node count; a canceled job with a recorded start remains a buggy executed
node.

The light image intentionally omits PyTorch, timm, and NVIDIA tools. Import
availability is supplied from the caller's Python environment; checks that
need a third-party model registry or `nvidia-smi` cannot be reproduced inside
this image. Do not claim those checks passed based on this container alone.
The two-epoch probe must therefore run in the candidate's own environment,
with real task fixtures and a bounded timeout. This feature is opt-in and
does not modify already running experiments.
