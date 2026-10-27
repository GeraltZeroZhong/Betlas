"""Equivalent parsing of independently constructed mmCIF records."""
import gzip
import pytest
from betlas.geometry_relations.extraction import parse_mmcif

def test_gemmi_preserves_biopython_mmcif_value_semantics():
    pytest.importorskip("gemmi")
    content = b'''data_synthetic
_entry.id toy
_note.text
;first line   
  second line  
;
loop_
_test.id
_test.value
1 ?
2 .
3 'two words'
4 "quoted value"
'''
    assert parse_mmcif(content, parser="gemmi") == parse_mmcif(content, parser="biopython")
    assert parse_mmcif(gzip.compress(content), parser="gemmi") == parse_mmcif(content, parser="biopython")
