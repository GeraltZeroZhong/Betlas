from __future__ import annotations

from betlas.io.cath import fold_label_from_cath


def test_cath_fold_label_priority() -> None:
    assert fold_label_from_cath("2.40.170.20", "TonB beta-barrel") == "beta_barrel"
    assert fold_label_from_cath("2.60.120.10", "Jelly Rolls") == "jelly_roll"
    assert fold_label_from_cath("2.60.40.10", "Immunoglobulins") == "beta_sandwich"
    assert fold_label_from_cath("2.90.10.10", "Orthogonal Prism") == "beta_prism"
    assert fold_label_from_cath("2.120.10.80", "Kelch-type beta propeller") == "beta_propeller"
    assert fold_label_from_cath("2.160.20.10", "3 Solenoid") == "beta_solenoid"
    assert fold_label_from_cath("3.20.20.10", "TIM Barrel") == "tim_like_beta_alpha_barrel"
