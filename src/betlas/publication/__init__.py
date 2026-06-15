"""Publication-oriented evidence generation for Betlas."""

from .figure_style import panel_label, publication_rc, save_figure, style_axis

__all__ = [
    "PublicationEvidenceResult",
    "panel_label",
    "publication_rc",
    "run_publication_evidence",
    "save_figure",
    "style_axis",
]


def __getattr__(name: str):
    if name in {"PublicationEvidenceResult", "run_publication_evidence"}:
        from .evidence import PublicationEvidenceResult, run_publication_evidence

        return {
            "PublicationEvidenceResult": PublicationEvidenceResult,
            "run_publication_evidence": run_publication_evidence,
        }[name]
    raise AttributeError(name)
