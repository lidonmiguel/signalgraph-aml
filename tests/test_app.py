from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_and_capacity_control_updates():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "IBM benchmark",
        "Behavior map",
        "Capacity planning",
        "Investigation queue",
        "Network explorer",
        "Methodology",
    ]
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Transactions benchmarked"] == "5,078,345"
    assert metric_values["Evaluation account-days"] == "720,800"
    assert metric_values["Precision @ 100"] == "20.0%"
    assert metric_values["Lift @ 100 over random"] == "43.94×"
    assert "Precision" in [metric.label for metric in app.metric]
    assert "Recall" in [metric.label for metric in app.metric]

    app.slider[0].set_value(110).run(timeout=45)
    assert not app.exception
    alert_capacity = next(metric for metric in app.metric if metric.label == "Alert capacity")
    assert alert_capacity.value == "110"
