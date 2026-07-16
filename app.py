"""Interactive SignalGraph AML investigation console."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from signalgraph_aml.data import generate_demo_transactions
from signalgraph_aml.evaluation import capacity_curve, evaluate_alerts, rank_alerts
from signalgraph_aml.features import build_account_day_features, temporal_train_mask
from signalgraph_aml.modeling import SignalGraphModel
from signalgraph_aml.profiling import build_cluster_profiles, cluster_name_map

st.set_page_config(
    page_title="SignalGraph AML",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at 12% 0%, #102942 0%, #07111f 38%);}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19,42,65,.96), rgba(10,26,43,.96));
        border: 1px solid rgba(76,230,180,.16); border-radius: 14px; padding: 17px;
        box-shadow: 0 8px 30px rgba(0,0,0,.18);
    }
    [data-testid="stSidebar"] {background: #081522; border-right: 1px solid #193149;}
    .signal-kicker {color:#39e6b0; font-size:.76rem; font-weight:700; letter-spacing:.18em;}
    .signal-title {font-size:2.35rem; font-weight:750; margin:.1rem 0 .25rem;}
    .signal-subtitle {color:#9db0c2; max-width:850px; margin-bottom:1.2rem;}
    .case-chip {display:inline-block; padding:.28rem .6rem; border-radius:99px;
        background:#123047; color:#7bf1ca; font-size:.78rem; margin-right:.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Learning normal account behavior…")
def build_demo_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = generate_demo_transactions()
    features = build_account_day_features(transactions)
    training = temporal_train_mask(features)
    model = SignalGraphModel(n_clusters=5).fit(features.loc[training])
    scored = model.score(features)
    profiles = build_cluster_profiles(scored)
    names = cluster_name_map(profiles)
    evaluation_cases = scored.loc[~training].copy()
    evaluation_cases["cluster_name"] = evaluation_cases["cluster"].map(names)
    return transactions, evaluation_cases, profiles


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def network_figure(
    transactions: pd.DataFrame,
    account: str,
    date: pd.Timestamp,
    reveal_labels: bool,
) -> go.Figure:
    day = transactions.loc[transactions["timestamp"].dt.normalize().eq(pd.Timestamp(date))]
    direct = day.loc[day["from_account"].eq(account) | day["to_account"].eq(account)]
    graph = nx.DiGraph()
    for row in direct.itertuples():
        previous = graph.get_edge_data(row.from_account, row.to_account, default={})
        graph.add_edge(
            row.from_account,
            row.to_account,
            weight=previous.get("weight", 0.0) + float(row.amount_paid),
            illicit=max(previous.get("illicit", 0), int(row.is_laundering)),
        )

    if graph.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No transactions found", showarrow=False)

    positions = nx.spring_layout(graph, seed=42, weight="weight", k=1.1)
    edge_traces: list[go.Scatter] = []
    for source, target, attributes in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        illicit = reveal_labels and attributes["illicit"] == 1
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 3 if illicit else 1.3, "color": "#ff5575" if illicit else "#395873"},
                hovertemplate=(
                    f"{source} → {target}<br>"
                    f"{money(attributes['weight'])}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in graph.nodes:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append("#39e6b0" if node == account else "#5c8fb7")
        node_size.append(25 if node == account else 13 + min(graph.degree(node) * 2, 10))
        node_text.append(f"{node}<br>{graph.degree(node)} connections")
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[account if node == account else "" for node in graph.nodes],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker={"size": node_size, "color": node_color, "line": {"width": 1, "color": "#b8fff0"}},
        showlegend=False,
    )
    figure = go.Figure(data=[*edge_traces, node_trace])
    figure.update_layout(
        height=510,
        margin={"l": 5, "r": 5, "t": 15, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


transactions, evaluation_cases, cluster_profiles = build_demo_state()

st.sidebar.markdown("### Monitoring controls")
max_budget = min(250, len(evaluation_cases))
alert_budget = st.sidebar.slider(
    "Daily investigation capacity", 10, max_budget, min(100, max_budget), 10
)
segment_options = {
    f"{int(row.cluster)} · {row.cluster_name}": int(row.cluster)
    for row in cluster_profiles.itertuples(index=False)
}
selected_segment_labels = st.sidebar.multiselect(
    "Behavioral segments",
    list(segment_options),
    default=list(segment_options),
)
reveal_labels = st.sidebar.toggle("Reveal ground truth", value=False)
st.sidebar.caption(
    "Ground truth is never used for clustering or anomaly scoring. "
    "Reveal it only to evaluate the demo."
)

selected_clusters = [segment_options[label] for label in selected_segment_labels]
filtered = evaluation_cases.loc[evaluation_cases["cluster"].isin(selected_clusters)].copy()
if filtered.empty:
    st.warning("Select at least one behavioral segment.")
    st.stop()

budget = min(alert_budget, len(filtered))
metrics = evaluate_alerts(filtered, alert_budget=budget)
alerts = rank_alerts(filtered, budget)

st.markdown('<div class="signal-kicker">FINANCIAL CRIME ANALYTICS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signal-title">SignalGraph '
    '<span style="color:#39e6b0">AML</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="signal-subtitle">An explainable, unsupervised investigation console '
    "that learns ordinary account behavior and prioritizes the cases most worth human "
    "review.</div>",
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Cases monitored", f"{metrics['cases']:,}")
metric_columns[1].metric("Alert capacity", f"{metrics['alert_budget']:,}")
metric_columns[2].metric("Precision @ K", f"{metrics['precision_at_k']:.1%}")
metric_columns[3].metric("Recall @ K", f"{metrics['recall_at_k']:.1%}")
metric_columns[4].metric("Lift over random", f"{metrics['lift_at_k']:.1f}×")

overview_tab, capacity_tab, queue_tab, network_tab, method_tab = st.tabs(
    [
        "Behavior map",
        "Capacity planning",
        "Investigation queue",
        "Network explorer",
        "Methodology",
    ]
)

with overview_tab:
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Behavioral landscape")
        plot_frame = filtered.copy()
        plot_frame["outcome"] = np.where(
            plot_frame["is_laundering"].eq(1), "Confirmed laundering", "Unconfirmed"
        )
        scatter_kwargs = {
            "data_frame": plot_frame,
            "x": "embedding_x",
            "y": "embedding_y",
            "color": "risk_score",
            "color_continuous_scale": ["#24445e", "#39e6b0", "#ffd166", "#ff5575"],
            "hover_name": "case_id",
            "hover_data": {
                "cluster": True,
                "cluster_name": True,
                "risk_score": True,
                "alert_reason": True,
                "embedding_x": False,
                "embedding_y": False,
            },
            "labels": {"risk_score": "Risk"},
        }
        if reveal_labels:
            scatter_kwargs["symbol"] = "outcome"
        scatter = px.scatter(**scatter_kwargs)
        scatter.update_traces(marker={"size": 8, "opacity": 0.76})
        scatter.update_layout(height=485, margin={"l": 5, "r": 5, "t": 10, "b": 5})
        st.plotly_chart(scatter, width="stretch")
    with right:
        st.subheader("Risk distribution")
        histogram = px.histogram(
            filtered,
            x="risk_score",
            nbins=24,
            color_discrete_sequence=["#39e6b0"],
            labels={"risk_score": "Risk score"},
        )
        histogram.add_vline(
            x=float(alerts["risk_score"].min()),
            line_dash="dash",
            line_color="#ffcf66",
            annotation_text="Alert threshold",
        )
        histogram.update_layout(
            height=300,
            showlegend=False,
            margin={"l": 5, "r": 5, "t": 10, "b": 5},
        )
        st.plotly_chart(histogram, width="stretch")
        st.markdown("**How to read this**")
        st.caption(
            "Each point is an account-day. Position summarizes behavior, color is model risk, "
            "and the dashed threshold reflects the current investigation capacity."
        )

    st.markdown("#### Behavioral segment profiles")
    st.caption(
        "Names describe median transaction behavior; they are not groups of suspects. "
        "The names are derived without laundering labels."
    )
    profile_table = cluster_profiles.copy()
    profile_table["median_value"] = profile_table["median_value"].map(money)
    profile_table["median_cross_currency_share"] = profile_table[
        "median_cross_currency_share"
    ].map(lambda value: f"{value:.0%}")
    profile_table["median_cash_share"] = profile_table["median_cash_share"].map(
        lambda value: f"{value:.0%}"
    )
    st.dataframe(
        profile_table,
        hide_index=True,
        width="stretch",
        column_config={
            "cluster": "Segment",
            "cluster_name": "Behavioral name",
            "cases": "Cases",
            "median_transactions": "Median transactions",
            "median_value": "Median value",
            "median_counterparties": "Median counterparties",
            "median_cross_currency_share": "Cross-currency share",
            "median_cash_share": "Cash share",
            "median_flow_ratio": "Flow balance",
        },
    )

with capacity_tab:
    st.subheader("Investigation capacity trade-off")
    st.caption(
        "Moving the capacity threshold does not retrain the model. It changes how many of the "
        "highest-ranked cases analysts review."
    )
    maximum_curve_capacity = min(250, len(filtered))
    capacities = sorted(
        set(range(10, maximum_curve_capacity + 1, 10)) | {min(budget, maximum_curve_capacity)}
    )
    curve = capacity_curve(filtered, capacities)
    tradeoff = go.Figure()
    tradeoff.add_trace(
        go.Scatter(
            x=curve["capacity"],
            y=curve["precision"],
            name="Precision",
            mode="lines+markers",
            line={"color": "#ffcf66", "width": 3},
        )
    )
    tradeoff.add_trace(
        go.Scatter(
            x=curve["capacity"],
            y=curve["recall"],
            name="Recall",
            mode="lines+markers",
            line={"color": "#39e6b0", "width": 3},
        )
    )
    tradeoff.add_vline(
        x=budget,
        line_dash="dash",
        line_color="#f4f7fb",
        annotation_text=f"Current capacity: {budget}",
    )
    tradeoff.update_layout(
        height=470,
        xaxis_title="Cases investigated (K)",
        yaxis_title="Metric",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.02],
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 5, "r": 5, "t": 60, "b": 5},
    )
    st.plotly_chart(tradeoff, width="stretch")
    selected_metrics = curve.loc[curve["capacity"].eq(min(budget, maximum_curve_capacity))].iloc[0]
    capacity_columns = st.columns(4)
    capacity_columns[0].metric("Cases reviewed", f"{int(selected_metrics.capacity):,}")
    capacity_columns[1].metric("Positive cases found", f"{int(selected_metrics.hits):,}")
    capacity_columns[2].metric("Precision", f"{selected_metrics.precision:.1%}")
    capacity_columns[3].metric("Recall", f"{selected_metrics.recall:.1%}")
    st.info(
        "A useful operating point balances missed suspicious cases against analyst workload. "
        "Higher capacity usually increases recall while reducing precision."
    )

with queue_tab:
    st.subheader("Ranked cases for analyst review")
    queue = alerts[
        [
            "case_id",
            "risk_score",
            "cluster",
            "cluster_name",
            "total_tx_count",
            "total_value",
            "unique_out_counterparties",
            "alert_reason",
            *( ["is_laundering"] if reveal_labels else [] ),
        ]
    ].copy()
    queue["total_value"] = queue["total_value"].map(money)
    st.dataframe(
        queue,
        hide_index=True,
        width="stretch",
        column_config={
            "case_id": "Case",
            "risk_score": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%.1f"
            ),
            "cluster": "Segment",
            "cluster_name": "Behavioral profile",
            "total_tx_count": "Transactions",
            "total_value": "Value moved",
            "unique_out_counterparties": "Payees",
            "alert_reason": "Primary explanation",
            "is_laundering": "Confirmed",
        },
        height=565,
    )

with network_tab:
    st.subheader("Account relationship explorer")
    selected_case = st.selectbox("Select an alert", alerts["case_id"].tolist())
    case = alerts.loc[alerts["case_id"].eq(selected_case)].iloc[0]
    st.markdown(
        f'<span class="case-chip">Risk {case.risk_score:.1f}</span>'
        f'<span class="case-chip">Segment {int(case.cluster)} · {case.cluster_name}</span>'
        f'<span class="case-chip">{money(case.total_value)} moved</span>',
        unsafe_allow_html=True,
    )
    graph_column, detail_column = st.columns([1.6, 1])
    with graph_column:
        st.plotly_chart(
            network_figure(transactions, case.account_id, case.date, reveal_labels),
            width="stretch",
        )
    with detail_column:
        st.markdown("#### Largest behavioral deviations")
        st.caption(
            "These factors explain how the case differs from its segment; they are not exact "
            "feature contributions or proof of laundering."
        )
        for factor in str(case.alert_factors).split(" • "):
            st.info(factor)
        st.metric("Transactions", f"{int(case.total_tx_count):,}")
        st.metric("Unique outgoing counterparties", f"{int(case.unique_out_counterparties):,}")
        cross_currency_count = int(round(case.cross_currency_share * case.out_tx_count))
        st.metric(
            "Cross-currency outgoing activity",
            f"{cross_currency_count} of {int(case.out_tx_count)} ({case.cross_currency_share:.0%})",
        )
        st.caption(
            "Nodes are accounts; lines aggregate transfers for the selected day. The selected "
            "account is green. If labels are revealed, known laundering edges are pink."
        )

with method_tab:
    st.subheader("Leakage-aware experimental design")
    st.markdown(
        """
        1. **Account-day aggregation** turns transaction logs into behavioral velocity, value,
           counterparty, bank-diversity, currency, and reciprocity features.
        2. **MiniBatch K-Means** discovers and names behavioral segments from the earliest 70%
           of dates without using laundering outcomes.
        3. **Cluster-relative Isolation Forests** identify unusual behavior inside each segment.
        4. **Operational ranking** combines anomaly percentile (82%) and distance from the
           segment center (18%).
        5. **Capacity analysis** shows the precision/recall trade-off at different analyst
           workloads.
        6. **Ground truth is revealed last** to calculate precision, recall, PR-AUC, and lift.
        """
    )
    st.warning(
        "This dashboard uses deterministic synthetic demo data. It demonstrates the product "
        "workflow; claims about model performance should use the IBM AML benchmark and "
        "documented validation."
    )
