from __future__ import annotations

from collections.abc import Iterable

TEXT = "#1F1F1F"
AXIS = "#3A3A3A"
GRID = "#E7E7E7"
LIGHT = "#FAFAF8"
PALE = "#ECEDEA"
GRAY = "#6E6E6E"
NEUTRAL = "#D6D6D2"
NEUTRAL_DARK = "#666666"
CHARCOAL = "#444444"
WHITE = "#FFFFFF"

# Nature-style, colorblind-accessible base hues derived from the
# Okabe-Ito/Wong palette and slightly muted for dense scientific panels.
BLACK = "#000000"
BLUE = "#0072B2"
SKY = "#56B4E9"
TEAL = "#009E73"
GREEN = "#009E73"
ORANGE = "#D55E00"
AMBER = "#E69F00"
YELLOW = "#F0E442"
PURPLE = "#7E68B3"
ROSE = "#CC79A7"
BROWN = "#8C6D31"
RED = "#D55E00"

SOURCE = RED
SINK = TEAL
SOURCE_LIGHT = "#E9A57C"
SINK_LIGHT = "#8CCFBE"

FOLD_COLORS: dict[str, str] = {
    "beta_barrel": ORANGE,
    "beta_prism": PURPLE,
    "beta_propeller": SKY,
    "jelly_roll": ROSE,
    "beta_solenoid": GREEN,
    "beta_sandwich": BLUE,
    "tim_like_beta_alpha_barrel": AMBER,
}

FOLD_DISPLAY_COLORS: dict[str, str] = {
    "Barrel": FOLD_COLORS["beta_barrel"],
    "Prism": FOLD_COLORS["beta_prism"],
    "Propeller": FOLD_COLORS["beta_propeller"],
    "Jelly roll": FOLD_COLORS["jelly_roll"],
    "Solenoid": FOLD_COLORS["beta_solenoid"],
    "Sandwich": FOLD_COLORS["beta_sandwich"],
    "TIM barrel": FOLD_COLORS["tim_like_beta_alpha_barrel"],
    "TIM-like": FOLD_COLORS["tim_like_beta_alpha_barrel"],
}

MODEL_COLORS: dict[str, str] = {
    "Betlas XGBoost": TEAL,
    "Betlas": TEAL,
    "Betlas+ESM-C": TEAL,
    "Betlas-Stave": TEAL,
    "Betlas-Stave-light": BLUE,
    "Geometry rules": GRAY,
    "Rules": GRAY,
    "ESM-C kNN, k=1": BLUE,
    "ESM-C kNN, k=5": "#6F96BF",
    "ESM-C kNN, k=10": "#9BB8D2",
    "Foldseek structural NN": AMBER,
    "Foldseek": AMBER,
    "Hist gradient boosting": SKY,
    "HistGB": SKY,
    "Logistic L2": PURPLE,
    "Logistic": PURPLE,
    "TMbed": AMBER,
    "PolarBearal3": PURPLE,
    "PROFtmb": GRAY,
}

MODEL_KEY_COLORS: dict[str, str] = {
    "xgboost_tuned": TEAL,
    "hist_gradient_boosting": SKY,
    "logistic_l2": PURPLE,
    "grammar_rules": GRAY,
}

READOUT_COLORS: dict[str, str] = {
    "sandwichness": BLUE,
    "jelly_rollness": ROSE,
    "barrel_likeness": ORANGE,
    "closure": ORANGE,
    "topology_ambiguity": CHARCOAL,
    "mixed_topology": GREEN,
    "source": SOURCE,
    "sink": SINK,
}

FEATURE_GROUP_COLORS: dict[str, str] = {
    "sequence_core": TEAL,
    "sheet_global": BLUE,
    "sheet_pair_packing": ROSE,
    "composition": GRAY,
    "beta_run_topology": AMBER,
    "axis_closure": SKY,
    "axis_periodicity": "#C99000",
    "angular_lobes": PURPLE,
    "alpha_shell": "#7A8E2F",
    "global_shape": "#607D8B",
    "strand_order": GREEN,
    "contact_graph": RED,
}

FLOW_COLORS: dict[str, str] = {
    "BB->BSand": "#D98D00",
    "BSand->BB": BLUE,
    "JR->BSand": ROSE,
    "BSand->JR": SKY,
    "BB->JR": ORANGE,
    "JR->BB": PURPLE,
}

STATUS_COLORS: dict[str, str] = {
    "valid numeric": TEAL,
    "missing/non-numeric": PALE,
    "failed": ORANGE,
    "baseline": TEAL,
    "edgerand": AMBER,
    "shuffle": NEUTRAL,
    "control": GRAY,
}

SEQUENTIAL_CMAPS: dict[str, list[str]] = {
    "blue": ["#FBFCFD", "#D9ECF7", "#8BC9E8", BLUE, "#004B78"],
    "green": ["#FBFCFA", "#DDF2EC", "#8FD8C6", GREEN, "#006B52"],
    "orange": ["#FFF9F4", "#F4D1BD", "#E49A65", ORANGE, "#8C3B00"],
    "rose": ["#FCF8FB", "#EFD8E7", "#DFA9CA", ROSE, "#8F4E78"],
    "purple": ["#FCF8FB", "#EFD8E7", "#DFA9CA", PURPLE, "#8F4E78"],
    "gray": ["#FCFCFC", "#E6E6E3", "#BDBDB8", GRAY, "#3D3D3D"],
    "distance": ["#FCFBF7", "#F3E3B4", "#E4BF54", AMBER, "#9B6500"],
    "density": ["#FCFBF7", "#DFF2ED", "#A6DFD0", "#4DBEAA", TEAL],
    "performance": ["#FCFBF7", "#DDF2EC", "#8FD8C6", GREEN, "#006B52"],
    "diverging_orange_blue": ["#8C3B00", "#E49A65", "#F7F7F2", "#8BC9E8", "#004B78"],
    "root": ["#DDD5C3", "#F0C36B", "#E58B4B", SOURCE],
    "sink": ["#DCE5E1", "#BFE6DC", SINK_LIGHT, SINK],
}


def color_for_fold(label: str, default: str = NEUTRAL_DARK) -> str:
    return FOLD_COLORS.get(str(label), default)


def color_for_fold_display(label: str, default: str = NEUTRAL_DARK) -> str:
    return FOLD_DISPLAY_COLORS.get(str(label), default)


def color_for_model(label: str, default: str = NEUTRAL_DARK) -> str:
    text = str(label)
    return MODEL_COLORS.get(text, MODEL_KEY_COLORS.get(text, default))


def color_for_readout(label: str, default: str = NEUTRAL_DARK) -> str:
    return READOUT_COLORS.get(str(label), default)


def rgb01(hex_color: str) -> list[float]:
    color = hex_color.strip().lstrip("#")
    return [int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def fold_palette(labels: Iterable[str]) -> dict[str, str]:
    fallback = [ORANGE, PURPLE, SKY, ROSE, GREEN, BLUE, AMBER, TEAL, BROWN]
    return {
        str(label): FOLD_COLORS.get(str(label), fallback[index % len(fallback)])
        for index, label in enumerate(labels)
    }


def cmap_colors(name: str) -> list[str]:
    return SEQUENTIAL_CMAPS.get(name, SEQUENTIAL_CMAPS["green"])


def matplotlib_cmap(name: str, *, lut: int = 256):
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(name, cmap_colors(name), N=lut)
