"""Public adapter API, loaded lazily so module CLIs execute only once."""

from importlib import import_module

_EXPORTS = {
    'BASELINE_NAME': ('runner', 'BASELINE_NAME'),
    'DEFAULT_ALIGNMENT_TYPE': ('runner', 'DEFAULT_ALIGNMENT_TYPE'),
    'DEFAULT_FORMAT_FIELDS': ('runner', 'DEFAULT_FORMAT_FIELDS'),
    'DEFAULT_MIN_QUERY_COVERAGE': ('runner', 'DEFAULT_MIN_QUERY_COVERAGE'),
    'DEFAULT_MIN_RESIDUES': ('structures', 'DEFAULT_MIN_RESIDUES'),
    'DEFAULT_MIN_TARGET_COVERAGE': ('runner', 'DEFAULT_MIN_TARGET_COVERAGE'),
    'DEFAULT_SCORE_MODE': ('runner', 'DEFAULT_SCORE_MODE'),
    'DEFAULT_SCORE_THRESHOLD': ('runner', 'DEFAULT_SCORE_THRESHOLD'),
    'FoldseekHit': ('runner', 'FoldseekHit'),
    'FoldseekResult': ('runner', 'FoldseekResult'),
    'GeneratedStructureChain': ('structures', 'GeneratedStructureChain'),
    'GeneratedStructureSet': ('structures', 'GeneratedStructureSet'),
    'StructureSearchBaselineRun': ('structure_search', 'StructureSearchBaselineRun'),
    'foldseek_query_aliases': ('structures', 'foldseek_query_aliases'),
    'generate_structure_chains': ('structures', 'generate_structure_chains'),
    'load_hits_tsv': ('runner', 'load_hits_tsv'),
    'run_baseline': ('runner', 'run_baseline'),
    'run_structure_search_baseline': ('structure_search', 'run_structure_search_baseline'),
    'summarize_hits': ('runner', 'summarize_hits'),
    'write_results_csv': ('runner', 'write_results_csv'),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), symbol)
    globals()[name] = value
    return value
