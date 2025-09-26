
# app.py — InnoRank-ARIE
# Streamlit app for the ARIE MCDM ranking method
# Author: ChatGPT (InnoRank-ARIE for IIDEX 2025 theme)
# ----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import json

# ----------------------------- Page & Theme ---------------------------------
st.set_page_config(
    page_title="InnoRank-ARIE",
    page_icon="🟣",
    layout="wide"
)

# Custom purple gradient theme (IIDEX-style)
st.markdown("""
<style>
:root {
  --primary:#7A2DF0;
  --accent:#C86BFF;
  --bg1:#0f0820;
  --bg2:#1b0f3a;
  --card:#160f2e;
  --text:#EDE7FF;
  --muted:#B9AEE6;
}
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
.block-container {
  padding-top: 2rem;
}
header {visibility:hidden;}
/* Gradient hero */
.hero {
  background: radial-gradient(1200px 400px at 20% -10%, rgba(200,107,255,0.25) 0%, rgba(200,107,255,0) 60%),
              radial-gradient(1200px 400px at 90% -10%, rgba(122,45,240,0.25) 0%, rgba(122,45,240,0) 60%),
              linear-gradient(180deg, var(--bg1), var(--bg2));
  border-radius: 18px;
  padding: 28px 28px 18px 28px;
  color: var(--text);
  border: 1px solid rgba(200,107,255,0.15);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 0 40px rgba(122,45,240,0.12);
}
h1.app-title { 
  font-size: 2.1rem; 
  letter-spacing: 0.3px; 
  margin: 0 0 4px 0;
}
.badge {
  display:inline-flex; align-items:center; gap:.4rem;
  background: rgba(122,45,240,.18);
  color: var(--text);
  padding:.25rem .6rem; border:1px solid rgba(200,107,255,.3);
  border-radius:999px; font-weight:600; font-size:.8rem;
}
.small { color: var(--muted); font-size:.9rem; }
.card {
  background: var(--card);
  border: 1px solid rgba(200,107,255,0.12);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
hr { border-color: rgba(200,107,255,0.18); }
table td, table th { color: var(--text)!important; }
.stAlert, .stDataFrame { background: transparent; }
.stButton>button {
  background: linear-gradient(90deg, var(--primary), var(--accent));
  color: white; border: 0; border-radius: 12px; padding: 0.55rem 1rem; font-weight: 700;
}
.stDownloadButton>button {
  border-radius: 12px; font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge">🧭 InnoRank‑ARIE • Adaptive Ranking with Ideal Evaluation</div>
  <h1 class="app-title">Intelligent MCDM Ranking System</h1>
  <div class="small">Upload your decision matrix, set criterion types & weights, and compute ARIE rankings with sensitivity controls (γ, κ).</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Helpers --------------------------------------
EPS = 1e-12

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Try converting all entries to float, preserving NaN for non-convertible."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def entropy_weights(X: pd.DataFrame) -> np.ndarray:
    """Compute objective weights via Shannon entropy (on positive normalized matrix)."""
    Xpos = X.clip(lower=EPS).astype(float)
    P = Xpos / (Xpos.sum(axis=0).replace(0, EPS))   # n x m, column-stochastic
    k = 1.0 / np.log(len(Xpos)) if len(Xpos) > 1 else 1.0
    E = -k * (P * np.log(P + EPS)).sum(axis=0)
    d = 1 - E
    w = d / d.sum() if d.sum() > 0 else np.full(shape=len(d), fill_value=1/len(d))
    return w.values if isinstance(w, pd.Series) else np.asarray(w)

def arie_normalize(X: pd.DataFrame, types: dict, targets: dict) -> pd.DataFrame:
    """Apply ARIE normalization per criterion type.
       types[col] in {"benefit","cost","target"}, targets[col] for 'target' only.
    """
    R = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for col in X.columns:
        x = X[col].astype(float)
        xmax = np.nanmax(x.values)
        xmin = np.nanmin(x.values)
        t = types[col]
        if t == "benefit":
            denom = xmax if np.abs(xmax) > EPS else EPS
            R[col] = x / denom
        elif t == "cost":
            R[col] = (xmin if np.abs(xmin) > EPS else EPS) / x.replace(0, EPS)
        elif t == "target":
            xT = float(targets.get(col, (xmax+xmin)/2 if np.isfinite(xmax) and np.isfinite(xmin) else 0.0))
            denom = max(abs(xmax - xT), abs(xmin - xT))
            denom = denom if denom > EPS else EPS
            R[col] = 1 - (x - xT).abs() / denom
        else:
            raise ValueError(f"Unknown type for column '{col}': {t}")
    return R

def arie_rank(X: pd.DataFrame, types: dict, weights: np.ndarray, targets: dict, gamma: float=1.0, kappa: float=0.5):
    """Compute ARIE ranking given X (n x m), types, weights (m,), targets, γ, κ."""
    R = arie_normalize(X, types, targets)                           # (2)-(4)
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be non-negative and sum to > 0.")
    w = w / w.sum()
    V = R * w                                                       # (5)
    v_max = V.max(axis=0)
    v_min = V.min(axis=0)

    # Similarity to ideal and anti-ideal (6)-(7) with safeguards
    Sim_best = ((V / (v_max.replace(0, EPS))).pow(gamma)).sum(axis=1)
    Sim_worst = (((v_min.replace(0, EPS)) / (V.replace(0, EPS))).pow(gamma)).sum(axis=1)

    # Relative closeness (8)
    RC = (kappa * Sim_best) / (kappa * Sim_best + (1 - kappa) * Sim_worst + EPS)

    # Assemble results
    df_sim = pd.DataFrame({
        "Sim_best": Sim_best,
        "Sim_worst": Sim_worst,
        "RC": RC
    })
    df_rank = (
        df_sim
        .sort_values(by="RC", ascending=False, kind="mergesort")
        .assign(Rank=lambda d: np.arange(1, len(d) + 1))
        .loc[:, ["Rank", "RC", "Sim_best", "Sim_worst"]]
    )
    df_rank.index.name = "Alternative"

    return {
        "R": R,
        "V": V,
        "v_max": v_max,
        "v_min": v_min,
        "sim": df_sim,
        "rank": df_rank
    }

def to_csv_download(df: pd.DataFrame, filename: str) -> bytes:
    # Ensure Alternative is a column and export minimal, RC-sorted ranking
    cols = ["Alternative", "Rank", "RC"]
    if df.index.name == "Alternative":
        out = df.reset_index()[cols]
    else:
        # Fallback if index is unnamed
        out = df.reset_index().rename(columns={"index":"Alternative"})[cols]
    return out.to_csv(index=False).encode("utf-8")

def to_excel_download(dfs: dict, filename: str="InnoRank_ARIE_results.xlsx") -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in dfs.items():
            # Always show Alternative as a column in Excel
            if hasattr(df.index, "name") and df.index.name is not None:
                df_to_write = df.reset_index()
            else:
                df_to_write = df
            df_to_write.to_excel(writer, sheet_name=name, index=False)
            if name == "Ranking":
                ws = writer.sheets[name]
                ws.freeze_panes(1, 0)
                try:
                    ws.set_column(0, 0, 18)  # Alternative
                    ws.set_column(1, 1, 8)   # Rank
                    ws.set_column(2, 4, 14)  # RC, Sim_best, Sim_worst
                except Exception:
                    pass
    output.seek(0)
    return output

# ----------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.write("Upload a **decision matrix** (rows = alternatives, columns = criteria).")
    up = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx","xls"])

    demo = st.toggle("Use demo dataset", value=(up is None))
    if demo:
        # small demo
        alts = ["A1","A2","A3","A4"]
        cols = ["C1","C2","C3","C4"]
        demo_data = pd.DataFrame(
            [[75,  8,  9.5, 30],
             [82, 12,  7.0, 26],
             [68, 10,  8.0, 28],
             [90,  6, 10.0, 24]],
            index=alts, columns=cols
        )
        st.caption("Demo meaning: C1 benefit, C2 cost, C3 target=9, C4 cost.")
    weight_mode = st.selectbox("Weights", ["Equal weights", "Manual", "Entropy (objective)"], index=0)
    gamma = st.slider("Sensitivity γ", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="γ>1 penalizes deviations more; γ<1 softens penalties.")
    kappa = st.slider("Balance κ", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="κ close to 1 emphasizes closeness to ideal; κ<0.5 emphasizes distance from worst.")
    st.markdown("---")
    st.markdown("**About ARIE**")
    st.caption("Normalization supports benefit, cost, and target (goal) types; then weighted similarities to ideal/nadir are computed and the Relative Closeness (RC) yields rankings.")

# ----------------------------- Data Load ------------------------------------
if up is not None:
    if up.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(up)
    else:
        df_raw = pd.read_excel(up)
    # If first column looks like labels, set index
    if df_raw.columns[0].lower() in ("alt","alternative","name","id","label"):
        df_raw = df_raw.set_index(df_raw.columns[0])
else:
    df_raw = demo_data.copy()

st.markdown("### 📊 Input Decision Matrix")
st.dataframe(df_raw, use_container_width=True)

# Validate and coerce
df_num = coerce_numeric_df(df_raw)
if df_num.isna().any().any():
    st.warning("Some values are non-numeric or missing. Non-numeric cells are set to NaN and will be ignored where possible. Please clean your data for best results.")
df_num = df_num.fillna(df_num.mean())

criteria = list(df_num.columns)
alts = list(df_num.index)

# ---------------------- Criterion Types & Targets ---------------------------
st.markdown("### 🧩 Criterion Settings")
types = {}
targets = {}
col_boxes = st.columns(len(criteria))
for i, c in enumerate(criteria):
    with col_boxes[i]:
        t = st.selectbox(f"{c} type", ["benefit","cost","target"], index=0, key=f"type_{c}")
        types[c] = t
        if t == "target":
            default_target = float(df_num[c].median())
            targets[c] = st.number_input(f"Target for {c}", value=default_target, key=f"target_{c}")
        else:
            if c in targets: targets.pop(c, None)

# ----------------------------- Weights --------------------------------------
st.markdown("### 🧮 Criterion Weights")
if weight_mode == "Equal weights":
    w = np.ones(len(criteria)) / len(criteria)
    st.write("Equal weights are applied (∑w = 1).")
elif weight_mode == "Entropy (objective)":
    # For entropy, we should normalize first to positive.
    # Use benefit-style pre-normalization (scale to column sums) to compute entropy objectively.
    try:
        w = entropy_weights(df_num.abs())
    except Exception as e:
        st.error(f"Entropy weight error: {e}")
        w = np.ones(len(criteria)) / len(criteria)
    st.write("Entropy-based objective weights computed from the data (∑w = 1).")
else:  # Manual
    w_inputs = []
    cols_w = st.columns(len(criteria))
    for i, c in enumerate(criteria):
        with cols_w[i]:
            w_inputs.append(st.number_input(f"w[{c}]", min_value=0.0, value=1.0, step=0.01, key=f"w_{c}"))
    w = np.array(w_inputs, dtype=float)
    if w.sum() <= 0:
        st.error("Manual weights must sum to > 0. Using equal weights temporarily.")
        w = np.ones(len(criteria)) / len(criteria)
w = w / w.sum()
df_w = pd.DataFrame({"weight": w}, index=criteria)
st.dataframe(df_w.T, use_container_width=True)

# ----------------------------- Run ARIE -------------------------------------
st.markdown("---")
run = st.button("🚀 Run ARIE Ranking", use_container_width=True)

if run:
    try:
        results = arie_rank(df_num, types, w, targets, gamma=gamma, kappa=kappa)
        R, V = results["R"], results["V"]
        v_max, v_min = results["v_max"], results["v_min"]
        df_sim, df_rank = results["sim"], results["rank"]

        st.success("ARIE computed successfully. Explore results below.")
        tab1, tab2, tab3, tab4 = st.tabs(["Normalized (R)", "Weighted (V)", "Similarities", "Ranking"])
        with tab1:
            st.dataframe(R.style.format("{:.6f}"), use_container_width=True)
        with tab2:
            st.dataframe(V.style.format("{:.6f}"), use_container_width=True)
            colv1, colv2 = st.columns(2)
            with colv1:
                st.markdown("**v<sub>j</sub><sup>max</sup>**")
                st.dataframe(pd.DataFrame(v_max).T, use_container_width=True)
            with colv2:
                st.markdown("**v<sub>j</sub><sup>min</sup>**")
                st.dataframe(pd.DataFrame(v_min).T, use_container_width=True)
        with tab3:
            st.dataframe(df_sim.style.format("{:.6f}"), use_container_width=True)
        with tab4:
            st.dataframe(df_rank.style.format({"RC":"{:.6f}", "Sim_best":"{:.6f}", "Sim_worst":"{:.6f}"}), use_container_width=True)
            rc_sorted = df_rank.reset_index()[["Alternative", "RC"]].set_index("Alternative")
            st.bar_chart(rc_sorted)

        # Downloads
        st.markdown("#### ⤵️ Download Results")
        excel_bytes = to_excel_download({
            "Input": df_num,
            "Normalized_R": R,
            "Weighted_V": V,
            "Similarities": df_sim,
            "Ranking": df_rank
        })
        st.download_button("Download Excel (all sheets)", data=excel_bytes, file_name="InnoRank_ARIE_results.xlsx")

        st.download_button("Download Ranking (CSV)", data=to_csv_download(df_rank, "ranking.csv"), file_name="InnoRank_ARIE_ranking.csv")

        # Export config as JSON
        config = {
            "criteria": criteria,
            "types": types,
            "targets": targets,
            "weights": {c: float(val) for c,val in zip(criteria, w)},
            "gamma": gamma,
            "kappa": kappa
        }
        st.download_button("Download Config (JSON)", data=json.dumps(config, indent=2).encode("utf-8"), file_name="InnoRank_ARIE_config.json")

    except Exception as e:
        st.error(f"Error while running ARIE: {e}")

# ----------------------------- Docs -----------------------------------------
with st.expander("📘 Method Notes (Formulas)"):
    st.markdown(r"""
**Normalization**
- Benefit: \( r_{ij} = \dfrac{x_{ij}}{\max_i x_{ij}} \)  
- Cost: \( r_{ij} = \dfrac{\min_i x_{ij}}{x_{ij}} \)  
- Target: \( r_{ij} = 1 - \dfrac{|x_{ij} - x_j^T|}{\max(|x_j^{\max}-x_j^T|, |x_j^{\min}-x_j^T|)} \)

**Weighting**  
\( v_{ij} = w_j \cdot r_{ij} \)

**Ideal & Anti-Ideal**  
\( v_j^{\max} = \max_i v_{ij}, \quad v_j^{\min} = \min_i v_{ij} \)

**Similarities (γ ≥ 0)**  
\( \mathrm{Sim}^{\text{best}}_i = \sum_{j=1}^n \left(\frac{v_{ij}}{v_j^{\max}}\right)^\gamma,\quad
   \mathrm{Sim}^{\text{worst}}_i = \sum_{j=1}^n \left(\frac{v_j^{\min}}{v_{ij}}\right)^\gamma \)

**Relative Closeness (κ ∈ [0,1])**  
\( RC_i = \dfrac{\kappa\cdot \mathrm{Sim}^{\text{best}}_i}{\kappa\cdot \mathrm{Sim}^{\text{best}}_i + (1-\kappa)\cdot \mathrm{Sim}^{\text{worst}}_i} \)

Rank alternatives by descending \( RC_i \).
""")
    st.caption("Safeguards: zero-division is avoided with a small epsilon.")

st.markdown("<div class='small'>© 2025 InnoRank‑ARIE • Built with Streamlit</div>", unsafe_allow_html=True)
