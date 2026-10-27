from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.external_baselines.embeddings import esmc


@pytest.mark.parametrize(('device', 'precision', 'flash', 'half'), [
    ('cpu', 'fp16', False, False),
    ('cpu', 'fp32', False, False),
    ('cuda', 'fp32', False, False),
    ('cuda', 'fp16', True, True),
])
def test_optional_attention_matches_device_and_precision(monkeypatch, tmp_path, device, precision, flash, half):
    import sys

    calls = {}

    class FakeModel:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def to(self, device):
            return self

        def load_state_dict(self, state, strict):
            assert state == {} and strict

        def half(self):
            calls['half'] = True
            return self

        def eval(self):
            pass

    modules = {name: ModuleType(name) for name in ['torch', 'esm', 'esm.models', 'esm.models.esmc', 'esm.tokenization']}
    modules['torch'].load = lambda *args, **kwargs: {}
    modules['torch'].cuda = SimpleNamespace(is_available=lambda: False)
    modules['esm.models.esmc'].ESMC = FakeModel
    modules['esm.tokenization'].EsmSequenceTokenizer = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    esmc.load_esmc_model(weights_path=tmp_path / 'fake.pth', device=device, precision=precision)
    assert calls['use_flash_attn'] is flash
    assert calls.get('half', False) is half


def test_structure_table_uses_csv_location_and_synthetic_chain(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'examples' / 'mini.cif'
    (tmp_path / 'mini.cif').write_bytes(fixture.read_bytes())
    table = tmp_path / 'structures.csv'
    table.write_text('record_id,structure_path,chain_id\ntoy_chain,mini.cif,A\n')
    records = esmc.load_structure_table(table)
    assert records.loc[0, 'structure_path'] == str(tmp_path / 'mini.cif')
    extracted = esmc.extract_sequences(records, min_residues=4)
    assert extracted.loc[0, 'record_id'] == 'toy_chain'
    assert extracted.loc[0, 'status'] == 'ok'
    assert extracted.loc[0, 'sequence_length'] == 16
    assert extracted.loc[0, 'chain_id'] == 'A'


def test_duplicate_record_ids_are_rejected(tmp_path):
    table = tmp_path / 'structures.csv'
    table.write_text('record_id,structure_path\na,first.cif\na,second.cif\n')
    with pytest.raises(ValueError, match='unique'):
        esmc.load_structure_table(table)


def test_alignment_preserves_missing_rows_and_input_order():
    matrix = np.array([[4.0, 5.0], [8.0, 9.0]], dtype=np.float32)
    metadata = pd.DataFrame({'record_id': ['third', 'first'], 'embedding_index': [0, 1]})
    aligned, available = esmc.align_embeddings(['first', 'missing', 'third'], matrix, metadata)
    np.testing.assert_array_equal(aligned, [[8, 9], [0, 0], [4, 5]])
    np.testing.assert_array_equal(available, [1, 0, 1])


def test_failed_sequences_require_no_model_load(tmp_path, monkeypatch):
    records = pd.DataFrame([{'record_id': 'absent', 'pdb_id': 'toy', 'filename': 'missing.cif', 'structure_path': str(tmp_path / 'missing.cif')}])
    sequences = esmc.extract_sequences(records, min_residues=4)
    assert sequences.loc[0, 'status'] == 'parse_error'
    monkeypatch.setattr(esmc, 'load_esmc_model', lambda **kwargs: pytest.fail('No model is needed without eligible sequences'))
    matrix, meta, _ = esmc.build_embeddings(sequences, weights_path=tmp_path / 'unused.pth', device='cpu', precision='fp32', max_length=10)
    assert matrix.shape == (0, 1152)
    assert list(meta.columns) == list(esmc.EMBEDDING_COLUMNS)


def test_first_embedding_index_remains_zero_in_sidecar(tmp_path, monkeypatch):
    table = tmp_path / 'records.csv'
    table.write_text('record_id,structure_path\ntoy,unused.cif\n')
    weights = tmp_path / 'fake-checkpoint.pth'
    weights.write_bytes(b'synthetic placeholder; backend is replaced in this test')
    monkeypatch.setattr(esmc, 'extract_sequences', lambda records, **kwargs: pd.DataFrame([{
        'record_id': 'toy', 'pdb_id': 'toy', 'filename': 'unused.cif', 'status': 'ok', 'sequence': 'AAAA',
        'chain_id': 'A', 'sequence_length': 4, 'reason': '',
    }]))
    monkeypatch.setattr(esmc, 'build_embeddings', lambda *args, **kwargs: (
        np.array([[2, 3]], dtype=np.float32),
        pd.DataFrame([{'record_id': 'toy', 'pdb_id': 'toy', 'filename': 'unused.cif', 'chain_id': 'A',
                       'sequence_length': 4, 'embedded_length': 4, 'truncated': 0, 'embedding_index': 0}]),
        {},
    ))
    target = tmp_path / 'out' / 'embeddings.npz'
    assert esmc.main(['--input-csv', str(table), '--weights', str(weights), '--out-npz', str(target)]) == 0
    metadata = pd.read_csv(target.parent / 'esmc_embedding_metadata.csv')
    assert metadata.loc[0, 'embedding_index'] == 0
    with np.load(target) as result:
        np.testing.assert_array_equal(result['embeddings'], [[2, 3]])
        assert result['record_id'].tolist() == ['toy']


def test_explicit_domain_sequences_keep_failures_and_never_read_structure(tmp_path):
    table = tmp_path / 'domains.csv'
    table.write_text('record_id,domain_sequence,structure_path,feature_status\n'
                     'valid,acdef,/absent/whole_chain.cif,no_beta_segments\n'
                     'missing,,/absent/whole_chain.cif,mapping_failed\n'
                     'invalid,A-CDE,/absent/whole_chain.cif,ok\n')
    records, sequences = esmc.load_sequence_table(table, sequence_column='domain_sequence', min_residues=4)
    assert records.record_id.tolist() == ['valid', 'missing', 'invalid']
    assert sequences.status.tolist() == ['ok', 'invalid_sequence', 'invalid_sequence']
    assert sequences.iloc[0].sequence == 'ACDEF'
    assert sequences.iloc[0].sequence_length == 5


def test_sequence_input_and_structure_input_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        esmc.parse_args(['--sequence-csv', 'a.csv', '--input-csv', 'b.csv', '--weights', 'x', '--out-npz', 'x.npz'])


def test_domain_length_policy_keeps_rejections_in_denominator():
    sequences = pd.DataFrame({'record_id': ['short', 'long', 'invalid'], 'sequence_length': [4, 8, 9],
                              'status': ['ok', 'ok', 'invalid_sequence'], 'reason': ['', '', 'bad input']})
    rejected = esmc.apply_length_policy(sequences, max_length=5, policy='reject')
    assert rejected.status.tolist() == ['ok', 'overlength', 'invalid_sequence']
    assert rejected.record_id.tolist() == sequences.record_id.tolist()
    truncated = esmc.apply_length_policy(sequences, max_length=5, policy='truncate')
    assert truncated.equals(sequences)


def test_nonfinite_embedding_is_not_reported_as_success(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext

    class Tensor:
        def __getitem__(self, key):
            return self

        def unsqueeze(self, _axis):
            return self

        def to(self, _device):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def mean(self, dim):
            return SimpleNamespace(numpy=lambda: np.array([np.nan, np.inf]))

    class Model:
        def encode(self, _protein):
            return SimpleNamespace(sequence=Tensor())

        def __call__(self, _tokens):
            return SimpleNamespace(embeddings=Tensor())

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'esm.sdk.api', SimpleNamespace(ESMProtein=lambda **kwargs: kwargs))
    monkeypatch.setattr(esmc, 'load_esmc_model', lambda **kwargs: (Model(), None, 'cpu'))
    sequences = pd.DataFrame([{'record_id': 'synthetic', 'sequence': 'AAAA', 'status': 'ok'}])
    with pytest.raises(FloatingPointError, match='record_id=synthetic'):
        esmc.build_embeddings(sequences, weights_path=tmp_path / 'unused', device='cpu', precision='fp32', max_length=10)
