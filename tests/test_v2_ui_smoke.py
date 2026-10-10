"""Exercise the real v2 Streamlit navigation/page with saved model artifacts."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_v2_streamlit_page_renders_real_models():
    from src.models.leakage_safe_v2 import PARAMS
    for domain in PARAMS:
        assert (ROOT / "models" / "v2" / f"{domain}.joblib").is_file()
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "3_Leakage_Safe_V2.py"), default_timeout=40)
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    assert len(app.selectbox) >= 2
    assert len(app.slider) >= 1
    assert any("Forecast" in str(x) for x in app.dataframe)
    # Exercise all available real domain models through the page controls.
    for choice in PARAMS:
        app.selectbox[0].set_value(choice).run()
        assert not app.exception, [e.message for e in app.exception]
        app.slider[0].set_value(min(2, PARAMS[choice]["horizon"])).run()
        assert not app.exception, [e.message for e in app.exception]
        for method in ["XGBoost only", "LightGBM only", "Calendar seasonal naive", "Moving average"]:
            app.selectbox[1].set_value(method).run()
            assert not app.exception, [e.message for e in app.exception]
