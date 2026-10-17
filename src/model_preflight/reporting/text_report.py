"""Human-readable terminal report."""

from model_preflight.core.results import PreflightReport


def render_text(report: PreflightReport) -> str:
    marker = " (GPU_CHECK_REQUIRED)" if report.gpu_check_required else ""
    lines = [
        f"Model Preflight: {report.overall_status.value}{marker}",
        f"Candidate: {report.candidate_id}",
        f"GPU submission recommended: {'yes' if report.gpu_submission_recommended else 'no'}",
        "",
        "Stages:",
    ]
    for stage in report.stages:
        lines.append(f"  {stage.status.value:14} {stage.name} ({stage.duration_seconds:.3f}s)")
        if any(
            item.get("shape_reduction", {}).get("applied", False)
            for item in stage.evidence.get("scenarios", [stage.evidence])
        ):
            lines.append(
                "                 reduced hidden widths; original numerical behavior untested"
            )
    if report.diagnostics:
        lines.extend(["", "Diagnostics:"])
        for item in report.diagnostics:
            location = ""
            if item.file:
                location = f" [{item.file}{':' + str(item.line) if item.line else ''}]"
            lines.append(f"  {item.severity.value.upper():7} {item.code}: {item.message}{location}")
    else:
        lines.extend(["", "No diagnostics."])
    lines.extend(
        [
            "",
            "A PASS validates configured CPU checks only; it does not prove CUDA correctness or",
            "GPU memory feasibility. Keep the target-GPU canary in the training job.",
        ]
    )
    return "\n".join(lines) + "\n"

