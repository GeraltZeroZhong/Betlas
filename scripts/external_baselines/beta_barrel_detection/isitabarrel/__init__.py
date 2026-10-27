"""Public adapter API, loaded lazily so module CLIs execute only once."""

from importlib import import_module

_EXPORTS = {
    'GeneratedContactMap': ('contact_maps', 'GeneratedContactMap'),
    'GeneratedContactMapSet': ('contact_maps', 'GeneratedContactMapSet'),
    'IsItABarrelResult': ('runner', 'IsItABarrelResult'),
    'StructureMapBaselineRun': ('structure_map', 'StructureMapBaselineRun'),
    'generate_structure_contact_maps': ('contact_maps', 'generate_structure_contact_maps'),
    'load_results_tsv': ('runner', 'load_results_tsv'),
    'run_baseline': ('runner', 'run_baseline'),
    'run_structure_map_baseline': ('structure_map', 'run_structure_map_baseline'),
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
