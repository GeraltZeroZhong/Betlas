from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import colors as cz_colors

MIN_ANNOTATION_SIZE = 5.5


def publication_rc(
    *,
    label_size: float = 7.0,
    title_size: float = 7.0,
    tick_size: float = 7.0,
    legend_size: float = 7.0,
    pad_inches: float = 0.035,
) -> dict[str, object]:
    """Matplotlib rcParams aligned to Nature-style editable figure output."""

    body_size = 7.0
    label_size = body_size
    title_size = body_size
    tick_size = body_size
    legend_size = body_size
    return {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": body_size,
        "font.weight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": cz_colors.AXIS,
        "axes.labelcolor": cz_colors.TEXT,
        "axes.linewidth": 0.55,
        "axes.labelsize": label_size,
        "axes.labelweight": "bold",
        "axes.titlesize": title_size,
        "axes.titleweight": "normal",
        "xtick.color": cz_colors.TEXT,
        "ytick.color": cz_colors.TEXT,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": legend_size,
        "legend.handlelength": 1.15,
        "legend.handletextpad": 0.35,
        "legend.columnspacing": 0.75,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "savefig.pad_inches": pad_inches,
        "text.color": cz_colors.TEXT,
    }


def style_axis(ax: Any, *, grid_axis: str | None = None, grid: bool = False) -> None:
    ax.tick_params(direction="out", width=0.55, length=2.3, color=cz_colors.AXIS)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(7.0)
    ax.yaxis.label.set_fontsize(7.0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("normal")
        label.set_fontsize(7.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(cz_colors.AXIS)
    if grid or grid_axis:
        axis = grid_axis if grid_axis in {"x", "y", "both"} else "both"
        ax.grid(True, axis=axis, color=cz_colors.GRID, linewidth=0.32, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: Any, label: str, *, x: float = -0.10, y: float = 1.045, size: float = 8.0) -> None:
    """Temporarily suppress in-artwork panel labels for pre-submission exports."""

    return None


def prepare_final_artwork(fig: Any, *, strip_titles: bool = True, strip_panel_labels: bool = True) -> None:
    """Apply final typography and remove transient titles/panel labels before export."""

    if strip_titles:
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            suptitle.set_text("")
        for ax in fig.axes:
            ax.set_title("")
    for ax in fig.axes:
        ax.xaxis.label.set_fontfamily("Arial")
        ax.yaxis.label.set_fontfamily("Arial")
        ax.xaxis.label.set_fontsize(7.0)
        ax.yaxis.label.set_fontsize(7.0)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontfamily("Arial")
            label.set_fontsize(7.0)
            label.set_fontweight("normal")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily("Arial")
                text.set_fontsize(7.0)
                text.set_fontweight("normal")
            if legend.get_title() is not None:
                legend.get_title().set_fontfamily("Arial")
                legend.get_title().set_fontsize(7.0)
                legend.get_title().set_fontweight("normal")
        for table in getattr(ax, "tables", []):
            for cell in table.get_celld().values():
                cell_text = cell.get_text()
                cell_text.set_fontfamily("Arial")
                cell_text.set_fontsize(min(7.0, max(MIN_ANNOTATION_SIZE, float(cell_text.get_fontsize()))))
                cell_text.set_fontweight("normal")
        for text in list(ax.texts):
            text.set_fontfamily("Arial")
            if strip_panel_labels and _looks_like_panel_heading(text, ax):
                text.remove()
            else:
                _clamp_annotation_text(text)
    for text in list(fig.texts):
        text.set_fontfamily("Arial")
        if strip_panel_labels and _looks_like_figure_heading(text):
            text.remove()
        else:
            _clamp_annotation_text(text)


def _clamp_annotation_text(text: Any) -> None:
    try:
        size = float(text.get_fontsize())
    except Exception:
        return
    text.set_fontweight("normal")
    text.set_fontsize(min(7.0, max(MIN_ANNOTATION_SIZE, size)))


def _looks_like_panel_heading(text: Any, ax: Any) -> bool:
    try:
        _x, y = text.get_position()
        transform = text.get_transform()
        weight = str(text.get_fontweight()).lower()
    except Exception:
        return False
    if transform != ax.transAxes or y < 0.97:
        return False
    if weight in {"bold", "heavy", "semibold", "demibold", "700", "800", "900"}:
        return True
    value = str(text.get_text()).strip()
    return len(value) == 1 and value.isalpha()


def _looks_like_figure_heading(text: Any) -> bool:
    try:
        _x, y = text.get_position()
        weight = str(text.get_fontweight()).lower()
    except Exception:
        return False
    return y >= 0.90 and weight in {"bold", "heavy", "semibold", "demibold", "700", "800", "900"}


def save_figure(
    fig: Any,
    base_path: Path,
    *,
    formats: Iterable[str] = ("png", "pdf", "tif"),
    png_dpi: int = 450,
    tif_dpi: int = 600,
    pad_inches: float = 0.035,
    bbox_inches: str = "tight",
    tiff_lzw: bool = True,
) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_final_artwork(fig)
    resolved_bbox, resolved_pad = _resolved_bbox(fig, bbox_inches=bbox_inches, pad_inches=pad_inches)
    outputs: list[Path] = []
    normalized = [suffix.strip(".").lower() for suffix in formats]
    for suffix in normalized:
        path = base_path.with_suffix(f".{suffix}")
        if suffix == "png":
            fig.savefig(path, dpi=png_dpi, bbox_inches=resolved_bbox, pad_inches=resolved_pad, facecolor="white")
        elif suffix == "tif" or suffix == "tiff":
            kwargs: dict[str, Any] = {}
            if tiff_lzw:
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(
                path,
                dpi=tif_dpi,
                bbox_inches=resolved_bbox,
                pad_inches=resolved_pad,
                facecolor="white",
                **kwargs,
            )
        elif suffix == "pdf":
            fig.savefig(
                path,
                bbox_inches=resolved_bbox,
                pad_inches=resolved_pad,
                facecolor="white",
                metadata={"Creator": "Betlas figure generator", "Producer": "Matplotlib", "CreationDate": None},
            )
        else:
            fig.savefig(path, bbox_inches=resolved_bbox, pad_inches=resolved_pad, facecolor="white")
        outputs.append(path)
    _normalize_rasters(outputs, png_dpi=png_dpi, tif_dpi=tif_dpi, tiff_lzw=tiff_lzw)
    return outputs


def _resolved_bbox(fig: Any, *, bbox_inches: str, pad_inches: float) -> tuple[Any, float]:
    if bbox_inches != "tight":
        return bbox_inches, pad_inches
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer).padded(pad_inches)
    except Exception:
        return bbox_inches, pad_inches
    return bbox, 0.0


def _normalize_rasters(paths: list[Path], *, png_dpi: int, tif_dpi: int, tiff_lzw: bool) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    for image_path in paths:
        suffix = image_path.suffix.lower()
        if suffix not in {".png", ".tif", ".tiff"}:
            continue
        dpi = (tif_dpi, tif_dpi) if suffix in {".tif", ".tiff"} else (png_dpi, png_dpi)
        with Image.open(image_path) as image:
            if image.mode == "RGBA":
                rgb = Image.new("RGB", image.size, "white")
                rgb.paste(image, mask=image.getchannel("A"))
            elif image.mode != "RGB":
                rgb = image.convert("RGB")
            else:
                rgb = image.copy()
            if suffix in {".tif", ".tiff"} and tiff_lzw:
                rgb.save(image_path, compression="tiff_lzw", dpi=dpi)
            else:
                rgb.save(image_path, dpi=dpi)
