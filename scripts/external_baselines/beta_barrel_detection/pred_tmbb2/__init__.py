"""Public adapter API, loaded lazily so module CLIs execute only once."""

from importlib import import_module

_EXPORTS = {
    'BASELINE_NAME': ('runner', 'BASELINE_NAME'),
    'DEFAULT_MIN_TM_STRANDS': ('runner', 'DEFAULT_MIN_TM_STRANDS'),
    'DEFAULT_PREDICTION_FIELD': ('runner', 'DEFAULT_PREDICTION_FIELD'),
    'GeneratedFastaSet': ('sequences', 'GeneratedFastaSet'),
    'GeneratedSequence': ('sequences', 'GeneratedSequence'),
    'PredTmbb2Result': ('runner', 'PredTmbb2Result'),
    'StructureSequenceBaselineRun': ('structure_sequence', 'StructureSequenceBaselineRun'),
    'generate_structure_fasta': ('sequences', 'generate_structure_fasta'),
    'parse_juchmme_stdout': ('runner', 'parse_juchmme_stdout'),
    'run_baseline': ('runner', 'run_baseline'),
    'run_structure_sequence_baseline': ('structure_sequence', 'run_structure_sequence_baseline'),
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
