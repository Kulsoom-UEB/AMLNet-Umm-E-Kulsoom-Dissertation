"""
Step 09: Streamlit Reviewer-Support Dashboard (polished).
CRISP-DM: Deployment (prototype).

Reads ONLY saved outputs produced by steps 06-08 (metrics tables, figures,
the frozen prediction file and explainability tables), so it starts instantly
and always shows exactly the numbers reported in the paper.

Run:
    streamlit run src/app.py

Requires: streamlit, plotly, pandas, numpy (see requirements.txt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC))
import amlnet_common as C  # noqa: E402

st.set_page_config(
    page_title="Explainable Machine Learning for Suspicious Transaction Risk Detection in AML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --navy:#12233f; --steel:#1f4e79; --blue:#4c72b0; --red:#c00000;
  --green:#2e7d32; --amber:#e6a700; --bg:#f4f6fb; --card:#ffffff;
}
html, body, [class*="css"] {
  font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1, h2, h3 { color: var(--navy); }

.hero {
  background: linear-gradient(90deg, #12233f 0%, #1f4e79 100%);
  border-radius: 14px; padding: 20px 26px; color: #fff; margin-bottom: 1.1rem;
}
.hero h1 { color: #fff; margin: 0; font-size: 1.55rem; font-weight: 700; }
.hero p  { color: #c9d6ea; margin: 6px 0 0; font-size: .92rem; }
.hero .hero-tag {
  display: inline-block; margin-top: 10px; padding: 3px 12px; border-radius: 999px;
  background: rgba(255,255,255,.14); color: #fff; font-size: .75rem; font-weight: 600;
}

.kpi-card {
  background: var(--card); border-radius: 12px; padding: 12px 16px;
  box-shadow: 0 1px 4px rgba(18,35,63,.08); border-left: 4px solid var(--blue);
  height: 100%;
}
.kpi-card .k-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: #5b6b85; }
.kpi-card .k-value { font-size: 1.4rem; font-weight: 700; color: var(--navy); margin-top: 2px; }
.kpi-card .k-sub   { font-size: .74rem; color: #8a97ad; }
.kpi-card.red   { border-left-color: var(--red); }
.kpi-card.green { border-left-color: var(--green); }
.kpi-card.amber { border-left-color: var(--amber); }

.badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600; }
.badge-High   { background:#fde8e8; color:var(--red);    border:1px solid #f5b5b5; }
.badge-Medium { background:#fff6dd; color:#8a6400;       border:1px solid #ecd48f; }
.badge-Low    { background:#e6f4ea; color:var(--green);  border:1px solid #b7ddc2; }

.card {
  background: var(--card); border-radius: 12px; padding: 16px 18px;
  box-shadow: 0 1px 4px rgba(18,35,63,.08); margin-bottom: .9rem;
}
.footer {
  text-align: center; color: #8a97ad; font-size: .78rem; margin-top: 2rem;
  border-top: 1px solid #e2e8f2; padding-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    p = C.TABLES_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def predictions() -> pd.DataFrame | None:
    p = C.DATA_PROCESSED_DIR / "test_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def config() -> dict | None:
    return (json.loads(C.FINAL_CONFIG_PATH.read_text())
            if C.FINAL_CONFIG_PATH.exists() else None)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to str so Streamlit/pyarrow never chokes."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df


def risk_bands(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Low / Medium / High bands derived from the operating threshold."""
    df = preds.copy()
    med = threshold * 0.5
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, med, threshold, 1.0001],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    return df


def badge(band: str) -> str:
    return f'<span class="badge badge-{band}">{band}</span>'


def band_icon(band: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(band, "⚪")


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi-card {tone}"><div class="k-label">{label}</div>'
            f'<div class="k-value">{value}</div><div class="k-sub">{sub}</div></div>')


def style_fig(fig, h: int = 360, title: str | None = None):
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        font=dict(family="Segoe UI, Arial", size=12, color="#12233f"),
        title=dict(text=title, font=dict(size=14, color="#12233f")) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def empty_state(msg: str):
    st.info(msg)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🛡️ Explainable Machine Learning for Suspicious Transaction Risk Detection in AML</h1>
  <p>Explainable machine-learning decision support for anti-money-laundering (AML) analysts, built on the synthetic AMLNet dataset. Academic prototype · human-review support only.</p>
  <span class="hero-tag">Decision support — not a legal determination</span>
</div>
""", unsafe_allow_html=True)

cfg = config()
if cfg is None:
    st.error("**No frozen model found.** Run the pipeline first: `python src/run_all.py` "
             "(or run the `AMLNet_RunAll.ipynb` notebook in Colab). "
             "Steps 00–08 must complete before the dashboard has anything to show.")
    st.stop()

default_t = float(cfg["selected_threshold"])
model_label = cfg["selected_model"].replace("_", " ")
strategy = cfg["imbalance_strategy"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Explainable ML for Suspicious Transaction Risk Detection in AML")
    st.caption("MSc Research Project · COM748")
    st.divider()
    st.markdown("**Selected model**")
    st.write(f"**{model_label}**")
    st.write(f"Imbalance strategy: {strategy}")
    st.write(f"Frozen threshold: **{default_t:.2f}**")
    st.caption("Model and threshold were selected automatically from validation model.")
    st.divider()
    threshold = st.slider(
        "Review threshold", 0.01, 0.99, default_t, 0.01,
        help="Transactions with risk score ≥ this value are treated as alerts.")
    st.divider()
    st.markdown(f"**Dataset**  \nAMLNet · Zenodo\n`{cfg.get('dataset_doi', '10.5281/zenodo.16736515')}`")
    st.caption("Synthetic data — results are controlled academic evidence, not "
               "proof of real-world deployment readiness.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
preds = predictions()
fin = table("final_test_metrics")
cm_tbl = table("final_test_confusion_matrix")
sel = table("model_selection_ranking")
tuning = table("hyperparameter_tuning_results")
val_test = table("validation_versus_test_generalisation")
shap_g = table("explainability_shap_importance_grouped")
builtin_g = table("explainability_builtin_importance_grouped")
perm = table("explainability_permutation_importance")
agree = table("explainability_method_agreement")
local_shap = table("explainability_local_shap_examples")
caveats = table("explainability_caveats")
sweep = table("selected_model_threshold_sweep")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
t_overview, t_queue, t_case, t_xai, t_profile, t_perf, t_limits = st.tabs(
    ["📊 Overview", "🚨 Alert queue", "🔍 Case review",
     "🧠 Explainability", "🎯 Risk profile", "📈 Performance", "⚠️ Limitations"]
)

# ============================ OVERVIEW ======================================
with t_overview:
    if fin is not None and len(fin):
        r = fin.iloc[0]
        alerts = int(r.get("predicted_positive_alerts", 0))
        tps = int(r.get("true_positives", 0))
        fpr = int(r.get("false_positives", 0))
        cols = st.columns(6)
        cards = [
            ("Model", model_label, strategy, ""),
            ("Threshold", f"{default_t:.2f}", "optimised on validation", "amber"),
            ("Precision", f"{r['precision']:.4f}", f"{tps} of {alerts} alerts genuine", "green"),
            ("Recall", f"{r['recall']:.4f}", f"{tps} of {tps + int(r.get('false_negatives',0))} positives", "green"),
            ("F1-score", f"{r['f1_score']:.4f}", "harmonic mean", ""),
            ("PR-AUC", f"{r['pr_auc_average_precision']:.4f}", "threshold-free", ""),
        ]
        for c, (lab, val, sub, tone) in zip(cols, cards):
            c.markdown(kpi(lab, val, sub, tone), unsafe_allow_html=True)
        st.markdown("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Test confusion matrix")
            if cm_tbl is not None:
                cm = cm_tbl.copy()
                cm.columns = [str(c) for c in cm.columns]
                # find the 2x2 numeric block
                try:
                    vals = cm.select_dtypes(include=[np.number]).values
                    if vals.size >= 4:
                        vals = vals[:2, :2] if vals.shape[0] >= 2 and vals.shape[1] >= 2 else vals
                        fig = go.Figure(go.Heatmap(
                            z=vals,
                            x=["Predicted normal", "Predicted laundering"],
                            y=["Actual normal", "Actual laundering"],
                            text=vals, texttemplate="%{text:,}",
                            colorscale="Blues", showscale=False,
                        ))
                        fig = style_fig(fig, 320, "Test confusion matrix")
                        st.plotly_chart(fig, width="stretch")
                    else:
                        empty_state("Confusion matrix table not in expected shape.")
                except Exception:
                    empty_state("Could not render confusion matrix from table.")
            else:
                empty_state("Run step 07 to generate the confusion matrix.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Precision–recall curve (test set)")
            if preds is not None and len(preds):
                from sklearn.metrics import precision_recall_curve
                prec, rec, _ = precision_recall_curve(preds["true_label"], preds["risk_score"])
                base = preds["true_label"].mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Model",
                                         line=dict(color="#1f4e79", width=2.5)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines",
                                         name=f"No-skill ({base:.4f})",
                                         line=dict(color="#8a97ad", dash="dash")))
                fig.add_trace(go.Scatter(x=[r["recall"]], y=[r["precision"]], mode="markers",
                                         name=f"Operating point (t={default_t:.2f})",
                                         marker=dict(color="#c00000", size=11, symbol="circle")))
                fig = style_fig(fig, 320, "Test precision–recall")
                st.plotly_chart(fig, width="stretch")
            else:
                empty_state("Run step 07 to generate test predictions.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Threshold optimisation (validation)")
        if sweep is not None and len(sweep):
            s = clean(sweep)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["threshold"], y=s["precision"], name="Precision",
                                     line=dict(color="#4c72b0", width=2)))
            fig.add_trace(go.Scatter(x=s["threshold"], y=s["recall"], name="Recall",
                                     line=dict(color="#2e7d32", width=2)))
            fig.add_trace(go.Scatter(x=s["threshold"], y=s["f1_score"], name="F1-score",
                                     line=dict(color="#c00000", width=2.5)))
            fig.add_vline(x=default_t, line_dash="dash", line_color="#12233f",
                          annotation_text=f" t*={default_t:.2f}")
            fig = style_fig(fig, 360, "Precision / recall / F1 across candidate thresholds")
            st.plotly_chart(fig, width="stretch")
        else:
            empty_state("Run step 06 to generate the threshold sweep.")
        st.markdown("</div>", unsafe_allow_html=True)

        if sel is not None and len(sel):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Automatic model selection (validation, optimised thresholds)")
            view = clean(sel)[["model_name", "imbalance_strategy", "optimised_threshold",
                               "precision", "recall", "f1_score",
                               "pr_auc_average_precision", "false_positives", "selected"]]
            view.columns = [c.replace("_", " ").title() for c in view.columns]
            st.dataframe(view, width="stretch", hide_index=True)
            # comparison chart
            comp = clean(sel).sort_values("pr_auc_average_precision")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=comp["model_name"], x=comp["pr_auc_average_precision"],
                                 name="PR-AUC", orientation="h",
                                 marker_color="#1f4e79"))
            fig.add_trace(go.Bar(y=comp["model_name"], x=comp["f1_score"],
                                 name="F1-score", orientation="h",
                                 marker_color="#8faadc"))
            fig = style_fig(fig, 380, "Model comparison (PR-AUC vs F1)")
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

# ============================ ALERT QUEUE ===================================
with t_queue:
    st.subheader("Prioritised alert queue")
    if preds is None:
        empty_state("Run step 07 to generate test predictions.")
    else:
        preds = risk_bands(preds, threshold)
        flagged = preds[preds["risk_band"] == "High"].copy()

        c = st.columns(4)
        c[0].metric("Transactions scored", f"{len(preds):,}")
        c[1].metric("Alerts (High risk)", f"{len(flagged):,}")
        c[2].metric("Alert rate", f"{100 * len(flagged) / len(preds):.3f} %")
        c[3].metric("Per 10k", f"{10000 * len(flagged) / len(preds):.1f}")

        st.markdown("**Risk-band distribution (all scored transactions)**")
        band_cols = st.columns(3)
        for i, band in enumerate(["Low", "Medium", "High"]):
            n = int((preds["risk_band"] == band).sum())
            pct = 100 * n / len(preds)
            tone = {"Low": "green", "Medium": "amber", "High": "red"}[band]
            band_cols[i].markdown(
                kpi(f"{band} risk", f"{n:,}", f"{pct:.2f}% of transactions", tone),
                unsafe_allow_html=True)

        st.markdown("**Alerts ranked by risk score**")
        band_filter = st.multiselect(
            "Filter by risk band", ["Low", "Medium", "High"],
            default=["High"], key="band_filter")
        subset = preds[preds["risk_band"].isin(band_filter)]
        show = subset.copy()
        show["band"] = show["risk_band"].map(band_icon)
        cols = [c for c in ["risk_score", "band", "outcome", "type", "category",
                            "amount", "oldbalanceOrg", "newbalanceOrig", "hour"]
                if c in show.columns]
        st.dataframe(clean(show.nlargest(300, "risk_score"))[cols],
                     width="stretch", height=400, hide_index=True)

        # Workload trade-off
        st.markdown("**Workload trade-off (test set, sweeping the threshold)**")
        rows = []
        for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
            p = (preds["risk_score"] >= t).astype(int)
            rows.append({"threshold": t,
                         "caught": int(((preds["true_label"] == 1) & (p == 1)).sum()),
                         "false alarms": int(((preds["true_label"] == 0) & (p == 1)).sum()),
                         "missed": int(((preds["true_label"] == 1) & (p == 0)).sum())})
        wt = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wt["threshold"], y=wt["caught"], name="Caught (TP)",
                                 line=dict(color="#2e7d32", width=2.5)))
        fig.add_trace(go.Scatter(x=wt["threshold"], y=wt["false alarms"], name="False alarms (FP)",
                                 line=dict(color="#c00000", width=2.5)))
        fig.add_trace(go.Scatter(x=wt["threshold"], y=wt["missed"], name="Missed (FN)",
                                 line=dict(color="#e6a700", width=2.5)))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#12233f",
                      annotation_text=f" t={threshold:.2f}")
        fig = style_fig(fig, 380, "Alert workload vs detection")
        st.plotly_chart(fig, width="stretch")

# ============================ CASE REVIEW ===================================
with t_case:
    st.subheader("Individual transaction review")
    if preds is None:
        empty_state("Run step 07 first.")
    else:
        preds = risk_bands(preds, threshold)
        pool = preds[preds["risk_band"] == "High"] if (preds["risk_band"] == "High").any() else preds
        if not len(pool):
            pool = preds
        options = pool.nlargest(100, "risk_score").index.tolist()
        pick = st.selectbox("Transaction to review", options, key="pick")
        row = preds.loc[[pick]]
        score = float(row["risk_score"].iloc[0])
        band = "High" if score >= threshold else ("Medium" if score >= threshold * 0.5 else "Low")
        c = st.columns(4)
        c[0].markdown(kpi("Risk score", f"{score:.4f}", "model probability", "red" if band == "High" else ""),
                      unsafe_allow_html=True)
        c[1].markdown(kpi("Risk band", band, badge(band), {"High": "red", "Medium": "amber", "Low": "green"}[band]),
                      unsafe_allow_html=True)
        c[2].metric("Above threshold", "Yes" if score >= threshold else "No")
        c[3].metric("Dataset label", row["outcome"].iloc[0] if "outcome" in row else "n/a")

        feat = [c for c in row.columns if c not in
                ("true_label", "risk_score", "predicted_label", "outcome", "risk_band")]
        st.markdown("**Transaction attributes**")
        st.dataframe(clean(row[feat].T.rename(columns={pick: "value"})),
                     width="stretch", hide_index=False)

        st.markdown("**Why this score? (local SHAP)**")
        if local_shap is not None and len(local_shap):
            match = local_shap[local_shap["row_index"] == pick]
            if len(match):
                m = match.sort_values("shap_contribution")
                fig = go.Figure(go.Bar(
                    y=m["feature_group"], x=m["shap_contribution"], orientation="h",
                    marker_color=["#c00000" if v > 0 else "#1f4e79" for v in m["shap_contribution"]],
                ))
                fig.add_vline(x=0, line_color="#12233f", line_width=0.8)
                fig = style_fig(fig, 320, "Local SHAP contributions")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(clean(m[["feature_group", "shap_contribution", "direction", "observed_value"]]),
                             width="stretch", hide_index=True)
            else:
                st.caption("Pre-computed explanation not stored for this row; "
                           "representative examples are shown below.")
                st.dataframe(clean(local_shap.head(15)), width="stretch", hide_index=True)
        else:
            empty_state("Run step 08 to generate local SHAP explanations.")

        st.markdown("**Reviewer actions (prototype — not persisted)**")
        rc1, rc2 = st.columns([2, 1])
        rc1.text_area("Reviewer notes", placeholder="Record observations about this transaction…")
        rc2.radio("Decision", ["Pending", "No further action", "Escalate"], horizontal=False)

# ============================ EXPLAINABILITY ================================
with t_xai:
    st.subheader("What drives the model?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Global SHAP importance (by feature group)**")
        if shap_g is not None and len(shap_g):
            d = clean(shap_g).sort_values("mean_abs_shap").tail(12)
            fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=d["feature_group"], orientation="h",
                                   marker_color="#1f4e79"))
            fig = style_fig(fig, 340, None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(clean(shap_g.head(12)), width="stretch", hide_index=True)
        else:
            empty_state("Run step 08 for SHAP importance.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Built-in importance (by feature group)**")
        if builtin_g is not None and len(builtin_g):
            d = clean(builtin_g).sort_values("importance_percent").tail(12)
            fig = go.Figure(go.Bar(x=d["importance_percent"], y=d["feature_group"], orientation="h",
                                   marker_color="#2e7d32"))
            fig = style_fig(fig, 340, None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(clean(builtin_g.head(12)), width="stretch", hide_index=True)
        else:
            empty_state("Run step 08 for built-in importance.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Permutation importance (drop in PR-AUC when shuffled)**")
    if perm is not None and len(perm):
        d = clean(perm).sort_values("permutation_importance_mean").tail(12)
        fig = go.Figure(go.Bar(
            x=d["permutation_importance_mean"], y=d["feature"], orientation="h",
            error_x=dict(type="data", array=d["permutation_importance_std"], visible=True),
            marker_color="#c00000"))
        fig = style_fig(fig, 380, None)
        st.plotly_chart(fig, width="stretch")
    else:
        empty_state("Run step 08 for permutation importance.")
    st.markdown("</div>", unsafe_allow_html=True)

    if agree is not None and len(agree):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Agreement between explanation methods**")
        st.dataframe(clean(agree.sort_values("shap_rank")), width="stretch", hide_index=True)
        st.caption("Ranks are computed per method; closer ranks mean stronger agreement.")
        st.markdown("</div>", unsafe_allow_html=True)

    if local_shap is not None and len(local_shap):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Local explanation examples**")
        st.dataframe(clean(local_shap.head(20)), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================ RISK PROFILE ==================================
with t_profile:
    st.subheader("Risk profile of flagged transactions")
    if preds is None:
        empty_state("Run step 07 first.")
    else:
        preds = risk_bands(preds, threshold)
        hi = preds[preds["risk_band"] == "High"]
        st.markdown(
            f"**{len(hi):,} of {len(preds):,} transactions "
            f"({100 * len(hi) / len(preds):.2f}%) fall in the High band "
            f"at threshold {threshold:.2f}.**")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Risk-band breakdown**")
            counts = preds["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.55, marker=dict(colors=["#2e7d32", "#e6a700", "#c00000"]),
                textinfo="label+percent"))
            fig = style_fig(fig, 320, None)
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Laundering-positive concentration by category**")
            if "category" in preds.columns:
                grp = (preds.groupby("category", observed=True)
                       .agg(total=("risk_score", "size"), positive=("true_label", "sum"))
                       .reset_index().sort_values("positive", ascending=False).head(12))
                fig = go.Figure(go.Bar(x=grp["positive"], y=grp["category"].astype(str),
                                       orientation="h", marker_color="#1f4e79"))
                fig = style_fig(fig, 320, None)
                st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Laundering-positive concentration by transaction type**")
        if "type" in preds.columns:
            grp = (preds.groupby("type", observed=True)
                   .agg(total=("risk_score", "size"), positive=("true_label", "sum"))
                   .reset_index().sort_values("positive", ascending=False))
            grp["positive_rate"] = 100 * grp["positive"] / grp["total"]
            st.dataframe(clean(grp), width="stretch", hide_index=True)
            fig = go.Figure(go.Bar(x=grp["type"].astype(str), y=grp["positive"],
                                   marker_color="#8faadc"))
            fig = style_fig(fig, 300, None)
            st.plotly_chart(fig, width="stretch")
        st.caption("High positive concentration in a few categories/types reflects the "
                   "structure of the synthetic generator (dataset limitation), not "
                   "generalisable laundering behaviour.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================ PERFORMANCE ===================================
with t_perf:
    st.subheader("Final test performance")
    if fin is not None and len(fin):
        r = fin.iloc[0]
        c = st.columns(4)
        c[0].metric("Precision", f"{r['precision']:.4f}")
        c[1].metric("Recall", f"{r['recall']:.4f}")
        c[2].metric("F1-score", f"{r['f1_score']:.4f}")
        c[3].metric("PR-AUC", f"{r['pr_auc_average_precision']:.4f}")
        c = st.columns(4)
        c[0].metric("True positives", int(r["true_positives"]))
        c[1].metric("False positives", int(r["false_positives"]))
        c[2].metric("False negatives", int(r["false_negatives"]))
        c[3].metric("ROC-AUC", f"{r['roc_auc']:.4f}")
        st.caption("Accuracy is not used for decisions: predicting 'normal' for "
                   "everything already scores ~99.84%.")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Final test metrics (full record)**")
        st.dataframe(clean(fin.T.reset_index().rename(columns={"index": "metric"})),
                     width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if val_test is not None and len(val_test):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Validation versus test (generalisation check)**")
            v = clean(val_test)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=v["metric"], y=v["validation"], name="Validation",
                                 marker_color="#4c72b0"))
            fig.add_trace(go.Bar(x=v["metric"], y=v["test"], name="Test",
                                 marker_color="#1f4e79"))
            fig = style_fig(fig, 320, None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(v, width="stretch", hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if tuning is not None and len(tuning):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Hyperparameter tuning results (RandomizedSearchCV, PR-AUC)**")
        st.dataframe(clean(tuning), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    p = C.FIGURES_DIR / "fig07_test_confusion_matrix.png"
    if p.exists():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Confusion matrix (pipeline figure)**")
        st.image(str(p), width=430)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================ LIMITATIONS ===================================
with t_limits:
    st.subheader("Limitations and responsible use")
    if caveats is not None and len(caveats):
        st.dataframe(clean(caveats), width="stretch", hide_index=True)
    st.markdown("""
- **AMLNet is synthetic.** The strong `category` / `type` signal is likely partly a
  generator artefact; performance does not transfer automatically to real data.
- **Leakage controls applied.** Typology labels, the generator `metadata` dictionary,
  `fraud_probability`, `isFraud`, account identifiers and the `step` simulation index
  were all excluded from the feature set.
- **Non-temporal split.** Train/validation/test used a *stratified random* split, not a
  chronological or account-disjoint one; results are evidence of pattern learning under
  controlled conditions, not guaranteed generalisation to future transactions or accounts.
- **Test set used once**, after the model and threshold were frozen on validation data.
- **Prototype only** — not evaluated with practising AML analysts; reviewer actions in
  this dashboard are not persisted.
""")

# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
  Explainable Machine Learning for Suspicious Transaction Risk Detection in AML · AMLNet Reviewer-Support Dashboard · Academic prototype. A high risk score means the transaction resembles
  patterns learned from the synthetic AMLNet dataset — it is not evidence, an
  accusation, or a legal determination. Every alert requires human review.
</div>
""", unsafe_allow_html=True)
