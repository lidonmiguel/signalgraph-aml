from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_and_capacity_control_updates():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Behavior map",
        "Capacity planning",
        "Investigation queue",
        "Network explorer",
        "Methodology",
    ]
    assert "Precision" in [metric.label for metric in app.metric]
    assert "Recall" in [metric.label for metric in app.metric]

    app.slider[0].set_value(110).run(timeout=45)
    assert not app.exception
    alert_capacity = next(metric for metric in app.metric if metric.label == "Alert capacity")
    assert alert_capacity.value == "110"
