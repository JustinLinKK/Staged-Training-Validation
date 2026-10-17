"""Conservative AST checks that never import candidate code."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from model_preflight.core.diagnostics import Diagnostic
from model_preflight.core.enums import Classification, Severity, StageName, StageStatus
from model_preflight.core.results import StageResult
from model_preflight.manifest.models import Manifest

_GPU_ONLY_IMPORTS = {
    "bitsandbytes",
    "flash_attn",
    "transformer_engine",
    "triton",
}
_DISTRIBUTED_IMPORTS = {"torch.distributed", "torch.distributed.fsdp"}
_TENSOR_FACTORIES = {"empty", "full", "ones", "rand", "randn", "zeros"}


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _diagnostic(
    code: str,
    message: str,
    path: Path,
    node: ast.AST | None,
    *,
    severity: Severity = Severity.WARNING,
    classification: Classification = Classification.RISK,
    operation: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=severity,
        stage=StageName.STATIC_SOURCE.value,
        classification=classification,
        message=message,
        file=str(path),
        line=getattr(node, "lineno", None),
        operation=operation,
        evidence=evidence or {},
        reproduction="model-preflight check preflight.yaml --only static_source",
    )


class _Visitor(ast.NodeVisitor):
    def __init__(
        self,
        display_path: Path,
        actual_path: Path,
        candidate_root: Path,
        batch_sizes: set[int],
    ) -> None:
        self.path = display_path
        self.actual_path = actual_path
        self.candidate_root = candidate_root
        self.batch_sizes = batch_sizes
        self.diagnostics: list[Diagnostic] = []

    def visit_Import(self, node: ast.Import) -> None:
        self._check_imports(node, [alias.name for alias in node.names])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._check_relative_import(node)
        if node.module:
            self._check_imports(node, [node.module])
        self.generic_visit(node)

    def _check_relative_import(self, node: ast.ImportFrom) -> None:
        if not (self.actual_path.parent / "__init__.py").is_file():
            exists = False
        else:
            base = self.actual_path.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                target = base.joinpath(*node.module.split("."))
                exists = target.with_suffix(".py").is_file() or (
                    target / "__init__.py"
                ).is_file()
            else:
                exists = all(
                    (base / alias.name).with_suffix(".py").is_file()
                    or (base / alias.name / "__init__.py").is_file()
                    for alias in node.names
                )
            try:
                base.resolve().relative_to(self.candidate_root)
            except ValueError:
                exists = False
        if not exists:
            self.diagnostics.append(
                _diagnostic(
                    "SRC002",
                    "relative import cannot be resolved within candidate.root",
                    self.path,
                    node,
                    severity=Severity.ERROR,
                    classification=Classification.CONFIRMED,
                    operation="relative_import",
                )
            )

    def _check_imports(self, node: ast.AST, names: list[str]) -> None:
        for imported in names:
            root = imported.split(".")[0]
            if root in _GPU_ONLY_IMPORTS or imported in _DISTRIBUTED_IMPORTS:
                self.diagnostics.append(
                    _diagnostic(
                        "GPU003",
                        (
                            f"{imported} may require CUDA and must be verified by "
                            "the target-GPU canary"
                        ),
                        self.path,
                        node,
                        classification=Classification.INCONCLUSIVE,
                        operation="import",
                        evidence={"import": imported},
                    )
                )

    def visit_Call(self, node: ast.Call) -> None:
        called = _name(node.func)
        if called == "torch.backends.cuda.is_available":
            self.diagnostics.append(
                _diagnostic(
                    "SRC_TORCH_API001",
                    (
                        "torch.backends.cuda has no is_available() API; use "
                        "torch.cuda.is_available() instead"
                    ),
                    self.path,
                    node,
                    severity=Severity.ERROR,
                    classification=Classification.CONFIRMED,
                    operation=called,
                )
            )
        if called.endswith(".cuda"):
            self.diagnostics.append(
                _diagnostic(
                    "DEV002",
                    "direct .cuda() call prevents device-independent CPU validation",
                    self.path,
                    node,
                    operation=called,
                )
            )
        if called == "torch.cuda.is_available":
            self.diagnostics.append(
                _diagnostic(
                    "GPU003",
                    "model behavior depends on torch.cuda.is_available(); verify the CUDA branch",
                    self.path,
                    node,
                    classification=Classification.INCONCLUSIVE,
                    operation=called,
                )
            )
        if called.startswith(
            ("torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.current_stream")
        ):
            self.diagnostics.append(
                _diagnostic(
                    "GPU003",
                    "CUDA stream behavior cannot be checked on CPU",
                    self.path,
                    node,
                    classification=Classification.INCONCLUSIVE,
                    operation=called,
                )
            )
        if called in {"torch.utils.cpp_extension.load", "torch.utils.cpp_extension.load_inline"}:
            self.diagnostics.append(
                _diagnostic(
                    "GPU003",
                    "custom extension compilation requires a separate build worker and GPU canary",
                    self.path,
                    node,
                    classification=Classification.INCONCLUSIVE,
                    operation=called,
                )
            )
        for keyword in node.keywords:
            if (
                keyword.arg == "device"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value.startswith("cuda")
            ):
                self.diagnostics.append(
                    _diagnostic(
                        "DEV001",
                        "hard-coded CUDA device prevents the configured CPU path",
                        self.path,
                        keyword.value,
                        operation=called,
                        evidence={"device": keyword.value.value},
                    )
                )
        if called.endswith((".view", ".reshape")) and node.args:
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, int)
                and first.value in self.batch_sizes
            ):
                self.diagnostics.append(
                    _diagnostic(
                        "BAT001",
                        "reshape/view appears to hard-code a configured batch size",
                        self.path,
                        first,
                        operation=called,
                        evidence={"literal_batch_size": first.value},
                    )
                )
        if called.startswith("torch.") and called.rsplit(".", 1)[-1] in _TENSOR_FACTORIES:
            self._check_factory_batch(node, called)
        self.generic_visit(node)

    def _check_factory_batch(self, node: ast.Call, called: str) -> None:
        if not node.args:
            return
        shape = node.args[0]
        first = (
            shape.elts[0]
            if isinstance(shape, (ast.Tuple, ast.List)) and shape.elts
            else shape
        )
        if (
            isinstance(first, ast.Constant)
            and isinstance(first.value, int)
            and first.value in self.batch_sizes
        ):
            self.diagnostics.append(
                _diagnostic(
                    "BAT001",
                    "tensor factory appears to hard-code a configured batch size",
                    self.path,
                    first,
                    operation=called,
                    evidence={"literal_batch_size": first.value},
                )
            )


def _entry_module_path(manifest: Manifest) -> Path:
    module_name = manifest.candidate.adapter.split(":", 1)[0]
    return manifest.candidate.root.joinpath(*module_name.split(".")).with_suffix(".py")


def _int_constants(tree: ast.AST) -> dict[str, int]:
    constants: dict[str, int] = {}
    for node in ast.walk(tree):
        value = None
        target = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            target, value = node.targets[0].id, node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            target, value = node.target.id, node.value.value
        if target is not None and value is not None:
            constants[target] = value
    return constants


def _moves_tensor_to_dynamic_device(node: ast.AST) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or _name(call.func).split(".")[-1] != "to":
            continue
        values = list(call.args) + [keyword.value for keyword in call.keywords]
        if any(
            isinstance(value, ast.Name) and value.id in {"device", "gpu_device"}
            for value in values
        ):
            return True
        if any(
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.startswith("cuda")
            for value in values
        ):
            return True
    return False


def _collate_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Lambda):
        return {
            child.id
            for child in ast.walk(node.body)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
    return set()


def _dataloader_cuda_worker_diagnostics(
    tree: ast.AST, path: Path
) -> list[Diagnostic]:
    constants = _int_constants(tree)
    cuda_collate_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _moves_tensor_to_dynamic_device(node)
    }
    diagnostics: list[Diagnostic] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _name(node.func).split(".")[-1] != "DataLoader":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        workers = keywords.get("num_workers")
        if isinstance(workers, ast.Name):
            worker_count = constants.get(workers.id)
        elif isinstance(workers, ast.Constant) and isinstance(workers.value, int):
            worker_count = workers.value
        else:
            worker_count = None
        collate = keywords.get("collate_fn")
        if (
            worker_count is not None
            and worker_count > 0
            and collate is not None
            and _collate_targets(collate) & cuda_collate_functions
        ):
            diagnostics.append(
                _diagnostic(
                    "SRC_DATALOADER_CUDA001",
                    "DataLoader workers invoke a collate function that moves tensors to CUDA; "
                    "keep collation on CPU and move the batch in the parent training loop",
                    path,
                    node,
                    severity=Severity.ERROR,
                    classification=Classification.CONFIRMED,
                    operation="DataLoader",
                    evidence={"num_workers": worker_count},
                )
            )
    return diagnostics


def run_static_checks(manifest: Manifest) -> StageResult:
    diagnostics: list[Diagnostic] = []
    entry_path = _entry_module_path(manifest)
    package_path = entry_path.with_suffix("") / "__init__.py"
    if not entry_path.is_file() and not package_path.is_file():
        diagnostics.append(
            Diagnostic(
                code="SRC002",
                severity=Severity.ERROR,
                stage=StageName.STATIC_SOURCE.value,
                classification=Classification.CONFIRMED,
                message=f"adapter module cannot be resolved under candidate.root: {entry_path}",
                file=str(entry_path),
                reproduction="model-preflight check preflight.yaml --only static_source",
            )
        )

    paths = sorted(
        path
        for path in manifest.candidate.root.rglob("*.py")
        if not any(part.startswith(".") for part in path.relative_to(manifest.candidate.root).parts)
    )
    for path in paths:
        display_path = path.relative_to(manifest.candidate.root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(display_path))
        except (OSError, UnicodeError) as exc:
            diagnostics.append(
                _diagnostic(
                    "SRC002",
                    f"cannot read Python source: {exc}",
                    display_path,
                    None,
                    severity=Severity.ERROR,
                    classification=Classification.CONFIRMED,
                )
            )
            continue
        except SyntaxError as exc:
            diagnostics.append(
                Diagnostic(
                    code="SRC001",
                    severity=Severity.ERROR,
                    stage=StageName.STATIC_SOURCE.value,
                    classification=Classification.CONFIRMED,
                    message=exc.msg,
                    file=str(display_path),
                    line=exc.lineno,
                    exception_type="SyntaxError",
                    evidence={"offset": exc.offset, "text": (exc.text or "").strip()},
                    reproduction="model-preflight check preflight.yaml --only static_source",
                )
            )
            continue
        visitor = _Visitor(
            display_path,
            path,
            manifest.candidate.root,
            set(manifest.scenarios.train_batch_sizes),
        )
        visitor.visit(tree)
        diagnostics.extend(visitor.diagnostics)
        diagnostics.extend(_dataloader_cuda_worker_diagnostics(tree, display_path))

    status = (
        StageStatus.FAIL
        if any(
            item.classification is Classification.CONFIRMED
            and item.severity is Severity.ERROR
            for item in diagnostics
        )
        else (
            StageStatus.INCONCLUSIVE
            if any(item.classification is Classification.INCONCLUSIVE for item in diagnostics)
            else StageStatus.PASS
        )
    )
    return StageResult(
        name=StageName.STATIC_SOURCE.value,
        status=status,
        diagnostics=diagnostics,
        evidence={"python_files_scanned": len(paths)},
    )
