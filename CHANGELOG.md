# Changelog

## 0.1.0 - Unreleased

- Add the standalone PyTorch preflight controller, isolated worker protocol, schemas, CLI,
  static checks, runtime checks, hardware profiles, and transparent memory-risk estimate.
- Require an isolated, CPU-only candidate probe image before speculative scheduler admission;
  use read-only source mounts and writable scratch, remove the exact probe container on timeout,
  and treat read-only fixture failures as inconclusive rather than candidate defects.

