import numpy as np
import pandas as pd

from betlas.geometry_relations.cohorts import A, H, T
from betlas.geometry_relations.packing import PackingScaler
from betlas.geometry_relations.scaffold_study import metadata_pool, sample_panel


def test_sampling_excludes_dependencies_and_records_missing_pools():
    frame=pd.DataFrame([
        {"record_id":f"d{i}","pdb_id":f"p{i//2}",A:"2.1" if i<10 else "2.2",
         T:"t1" if i%2 else "t2",H:f"h{i//3}","observed_length":100+i,
         "beta_fraction":.5,"strand_count":8,"missing_fraction":0,"x":float(i)}
        for i in range(16)
    ])
    scaler=PackingScaler.fit(frame,{"shape":["x"]},apply_applicability=False)
    anchors,pairs=sample_panel(frame,scaler)
    assert len(anchors)==8 and len(pairs)<=40
    assert pairs.query_cath_homology_code.ne(pairs.target_cath_homology_code).all()
    assert pairs.query_pdb_id.ne(pairs.target_pdb_id).all()
    assert not pairs.duplicated(["anchor_id","target_id"]).any()
    a2,p2=sample_panel(frame,scaler)
    pd.testing.assert_frame_equal(anchors,a2)
    pd.testing.assert_frame_equal(pairs,p2)
    singleton=frame.iloc[:1].copy()
    a,p=sample_panel(singleton,scaler)
    assert a.iloc[0].status=="no_matched_targets" and p.empty


def test_missing_metadata_cannot_enter_matching_pool():
    frame=pd.DataFrame({"record_id":["q","ok","missing"],"pdb_id":["p","r","s"],
                        H:["h1","h2","h3"],"observed_length":[100,100,100],
                        "beta_fraction":[.5,.5,np.nan],"strand_count":[8,8,8],
                        "missing_fraction":[0,0,0]})
    assert metadata_pool(frame,frame.iloc[0]).tolist()==[False,True,False]
