import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_survival_row(**kwargs):
    """Make a minimal survival CSV row with defaults."""
    defaults = dict(
        eid=12345, sex=1, bmi=28.4, smoking_status=1, alcohol_status=1,
        living_alone=0, hba1c=42.0, hdl_cholesterol=1.1, total_cholesterol=5.6,
        c_reactive_protein=3.2, triglycerides=1.8, glucose=5.1, albumin=44.0,
        creatinine=95.0, urea=6.5, total_protein=72.0, bilirubin=12.0, alt=28.0,
        haemoglobin=145.0, mcv=88.0, alkaline_phosphatase=75.0,
        apolipoprotein_a=1.5, apolipoprotein_b=0.9, cystatin_c=0.85,
        igf1=18.2, lipoprotein_a=25.0, systolic_variation=8.5,
        diabetes_t1=0, diabetes_t2=1, ckd=0, depression=0,
    )
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_biomarker_text_contains_demographics():
    from preprocessing.natural_text_conversion import BiomarkerBefore60TextConverter
    conv = BiomarkerBefore60TextConverter.__new__(BiomarkerBefore60TextConverter)
    row = _make_survival_row(sex=1, bmi=28.4, smoking_status=1)
    disease_text = "At age 30.0, patient was diagnosed with J45 asthma."
    text = conv._format_biomarker_text(row, disease_text)
    assert "male" in text.lower() or "Male" in text
    assert "28.4" in text


def test_biomarker_text_contains_labs():
    from preprocessing.natural_text_conversion import BiomarkerBefore60TextConverter
    conv = BiomarkerBefore60TextConverter.__new__(BiomarkerBefore60TextConverter)
    row = _make_survival_row(hba1c=48.5, hdl_cholesterol=1.2)
    text = conv._format_biomarker_text(row, "No diseases diagnosed before age 60.")
    assert "48.5" in text
    assert "1.2" in text


def test_biomarker_text_contains_disease_history():
    from preprocessing.natural_text_conversion import BiomarkerBefore60TextConverter
    conv = BiomarkerBefore60TextConverter.__new__(BiomarkerBefore60TextConverter)
    row = _make_survival_row()
    disease_text = "At age 45.0, patient was diagnosed with J45 asthma."
    text = conv._format_biomarker_text(row, disease_text)
    assert "45.0" in text
    assert "J45" in text


def test_biomarker_text_no_nan_in_output():
    from preprocessing.natural_text_conversion import BiomarkerBefore60TextConverter
    conv = BiomarkerBefore60TextConverter.__new__(BiomarkerBefore60TextConverter)
    row = _make_survival_row(hba1c=float("nan"), creatinine=float("nan"))
    text = conv._format_biomarker_text(row, "No diseases diagnosed before age 60.")
    assert "nan" not in text.lower()
    assert "NaN" not in text
