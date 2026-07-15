import pandas as pd
import streamlit as st


def inject_css():
    st.markdown(
        """
<style>
    .stApp { background: #0b0d10; color: #f4f6f8; }
    [data-testid="stSidebar"] { background: #11151b; }
    h1, h2, h3 { color: #f4f6f8; letter-spacing: 0; }
    .race-card {
        border: 1px solid #252b33;
        background: linear-gradient(135deg, #151a21, #0e1116);
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: #e10600;
        color: white;
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
    }
    .soft-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: #252b33;
        color: #d8dde5;
        font-size: 12px;
        margin-right: 8px;
    }
    .metric-card {
        border: 1px solid #252b33;
        background: #121720;
        padding: 14px;
        border-radius: 8px;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(f"# {title}")
    st.markdown(f"<span style='color:#aeb7c2'>{subtitle}</span>", unsafe_allow_html=True)


def race_summary(meta):
    sprint = "SPRINT" if meta.get("sprint_weekend") else "STANDARD"
    st.markdown(
        f"""
<div class="race-card">
  <span class="badge">{meta.get('prediction_mode')}</span>
  <span class="soft-badge">{sprint}</span>
  <span class="soft-badge">Round {meta.get('round')}</span>
  <h3>{meta.get('raceName')}</h3>
  <p style="color:#aeb7c2">Race date: {meta.get('date')} · Cutoff: {meta.get('data_cutoff')} · Model: {meta.get('model_version')}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def prediction_table(predictions):
    if predictions.empty:
        st.warning("No predictions available.")
        return
    show = predictions.copy()
    for col in ["proba_top10", "proba_podium"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    show["expected_points"] = show["expected_points"].round(2)
    show["race_rank_score"] = show["race_rank_score"].round(3)
    st.dataframe(
        show[
            [
                "predicted_rank", "driver_name", "constructor_name",
                "proba_top10", "proba_podium", "expected_points",
                "race_rank_score", "confidence",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def context_block(meta, future_inputs):
    cols = st.columns(4)
    cols[0].metric("Drivers available", meta.get("drivers_available", 0))
    cols[1].metric("Prediction mode", meta.get("prediction_mode", "-"))
    cols[2].metric("Confidence", meta.get("confidence_level", "-"))
    cols[3].metric("Missing features", len(meta.get("missing_features", [])))
    if meta.get("missing_features"):
        st.caption("Missing model features were added as NaN and handled by the saved preprocessing pipeline.")
        st.write(meta["missing_features"])
    if isinstance(future_inputs, pd.DataFrame) and not future_inputs.empty:
        st.dataframe(future_inputs.head(30), use_container_width=True, hide_index=True)
