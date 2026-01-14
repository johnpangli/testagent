import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import re
import us
from census import Census
import google.generativeai as genai
from pypdf import PdfReader

# =============================================================================
# 1) PAGE CONFIG + CLEAN LIGHT UI
# =============================================================================
st.set_page_config(page_title="Product Intelligence Hub", page_icon="■", layout="wide")

st.markdown("""
<style>
/* ----------------- Global typography ----------------- */
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
div.block-container { padding-top: 3.25rem; max-width: 1400px; }

/* ----------------- Force app chrome to light gray ----------------- */
/* Top header bar (Streamlit) */
header[data-testid="stHeader"] {
  background: #f6f7fb !important;
  border-bottom: 1px solid #e5e7eb !important;
}
/* Toolbar row (sometimes appears as a dark strip) */
div[data-testid="stToolbar"] {
  background: #f6f7fb !important;
}
/* Hide Streamlit deploy/menu/footer clutter (optional but cleaner) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
div[data-testid="stDecoration"] { background: #f6f7fb !important; }

/* App background */
.stApp {
  background: #f6f7fb;
  color: #0f172a;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
  background: #eef2f7 !important;
  border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * {
  color: #0f172a !important;
}

/* Headings */
h1 { font-size: 24px !important; margin: 0.1rem 0 0.25rem 0; letter-spacing: -0.02em; }
h2 { font-size: 16px !important; margin: 0.85rem 0 0.25rem 0; letter-spacing: -0.02em; }
h3 { font-size: 14px !important; margin: 0.65rem 0 0.25rem 0; letter-spacing: -0.02em; }

.small-muted { color: #64748b !important; font-size: 12px; line-height: 1.35; }
.hr { height: 1px; background: #e5e7eb; margin: 12px 0; }

/* ----------------- Buttons (tight + premium) ----------------- */
.stButton>button {
  width: 100%;
  border-radius: 12px !important;
  font-weight: 650 !important;
  padding: 0.5rem 0.85rem !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  color: #0f172a !important;
}
.stButton>button:hover {
  border-color: #cbd5e1 !important;
  background: #fbfbfd !important;
}
button[kind="primary"] {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
}
button[kind="primary"]:hover {
  background: #f8fafc !important;
  border-color: #94a3b8 !important;
}

/* Reduce vertical spacing between blocks */
div[data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* ----------------- Cards / Tiles ----------------- */
.card {
  background: #ffffff;
  border: 1px solid #e6e8ef;
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 1px 0 rgba(15,23,42,0.04);
}
.card-title {
  font-size: 13.5px;
  font-weight: 780;
  color: #0f172a;
  margin-bottom: 2px;
}
.card-sub {
  font-size: 12px;
  color: #64748b;
  margin-bottom: 10px;
}
.card-bullets {
  font-size: 13px;
  color: #0f172a;
  line-height: 1.35;
}
.card-bullets div { margin: 6px 0; }
.tight-controls { margin-top: -6px; } /* pulls buttons up closer */

/* Section label (subtle) */
.section-label {
  font-size: 11.5px;
  font-weight: 800;
  color: #334155;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 10px 0 6px 2px;
}

/* ----------------- Tables (light) ----------------- */
table { width: 100%; border-collapse: collapse; background: #ffffff; }
th {
  background: #f8fafc;
  color: #0f172a !important;
  text-align: left;
  padding: 10px;
  font-weight: 800;
  border-bottom: 1px solid #e5e7eb;
}
td {
  padding: 9px 10px;
  color: #0f172a;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: top;
}

/* Expander summary + body (force light) */
details {
  background: #ffffff !important;
  border: 1px solid #e6e8ef !important;
  border-radius: 14px !important;
  padding: 6px 10px !important;
}
details summary {
  color: #0f172a !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2) SESSION STATE
# =============================================================================
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "demographics_df" not in st.session_state:
    st.session_state.demographics_df = None
if "trends_text" not in st.session_state:
    st.session_state.trends_text = ""

# UI state
if "active_panel" not in st.session_state:
    st.session_state.active_panel = None  # "market" | "occasions" | "claims" | "ingredients" | "exec"
if "directive_result" not in st.session_state:
    st.session_state.directive_result = None
if "ui_locked" not in st.session_state:
    st.session_state.ui_locked = False

# Persist selections across reruns
if "sel_region" not in st.session_state:
    st.session_state.sel_region = "Midwest"
if "sel_category" not in st.session_state:
    st.session_state.sel_category = "Snack Nuts"
if "sel_focus" not in st.session_state:
    st.session_state.sel_focus = None
if "sel_competitors" not in st.session_state:
    st.session_state.sel_competitors = []
if "show_focus_block" not in st.session_state:
    st.session_state.show_focus_block = True
if "show_top_entities" not in st.session_state:
    st.session_state.show_top_entities = True

# =============================================================================
# 3) MDM ENTITY RESOLUTION (LOCKED TO GEMINI-3-FLASH-PREVIEW)
# =============================================================================
def get_canonical_parent_map(messy_brands, api_key):
    if not messy_brands or not api_key:
        return {}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")  # LOCKED

    prompt = f"""
ACT AS: Enterprise Master Data Management (MDM) Specialist for a CPG Firm.
TASK: Clean this list of messy brand strings and map them to their ONE true Parent Company.

RULES:
1) CONSOLIDATE VARIATIONS (e.g., "Blue Diamond", "Blue Diamond Almonds" -> "Blue Diamond Growers").
2) RESOLVE PARENTS (e.g., "Wright" -> "Tyson Foods").
3) RETAILER BRANDS (e.g., "365" -> "Amazon/Whole Foods").
4) PRIVATE LABEL (e.g., "Great Value" -> "Walmart", "Kirkland" -> "Costco").
5) HIERARCHY: aim for ultimate corporate owner.

LIST:
{messy_brands}

RETURN JSON ONLY:
{{
  "Mapping": [
    {{"raw": "Messy Name", "canonical_parent": "Clean Parent Company"}}
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r"```json\s?|```", "", response.text).strip()
        data = json.loads(clean_json)
        return {item["raw"]: item["canonical_parent"] for item in data.get("Mapping", [])}
    except Exception as e:
        st.error(f"MDM Engine Error: {e}")
        return {b: b for b in messy_brands}

# =============================================================================
# 4) DATA ACQUISITION
# =============================================================================
REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA"],
    "South": ["TX", "FL", "GA", "NC", "VA"],
    "West": ["CA", "WA", "AZ", "CO"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH"],
}

CATEGORY_MAP = {
    "Bacon": "bacons",
    "Peanut Butter": "peanut-butters",
    "Snack Nuts": "nuts",
    "Beef Jerky": "meat-snacks",
    "Coffee": "coffees",
    "Cereal": "breakfast-cereals",
    "Chips": "chips",
}

def fetch_market_intelligence(category, gemini_key):
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    headers = {"User-Agent": "ProductIntelligenceHub/1.0"}
    all_products = []
    status_text = st.empty()

    for page in range(1, 6):
        status_text.text(f"Scanning products · page {page}")
        url = (
            "https://world.openfoodfacts.org/cgi/search.pl?"
            "action=process"
            f"&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}"
            "&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States"
            "&json=1&page_size=100"
            f"&page={page}"
            "&fields=product_name,brands,countries_tags,ingredients_text,labels_tags,unique_scans_n"
        )
        try:
            r = requests.get(url, headers=headers, timeout=15)
            products = r.json().get("products", [])
            if not products:
                break
            all_products.extend(products)
            time.sleep(0.45)
        except:
            break

    status_text.empty()
    df = pd.DataFrame(all_products)
    if df.empty:
        return df

    df["brands"] = df["brands"].astype(str).str.strip().str.strip(",")
    df = df[~df["brands"].isin(["nan", "None", "", "Unknown", "null"])]
    df = df.drop_duplicates(subset=["product_name"])

    unique_messy = df["brands"].unique().tolist()
    with st.spinner(f"Normalizing entities ({len(unique_messy)} brands)…"):
        parent_map = get_canonical_parent_map(unique_messy, gemini_key)

    df["parent_company"] = df["brands"].map(parent_map).fillna(df["brands"])
    df["unique_scans_n"] = pd.to_numeric(df.get("unique_scans_n"), errors="coerce").fillna(0)
    return df

def fetch_demographics(census_key, region):
    if not census_key:
        return None
    c = Census(census_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []
    vars = ("B01003_001E", "B19013_001E", "B17001_002E", "B17001_001E")

    for s_code in states:
        try:
            state_obj = us.states.lookup(s_code)
            res = c.acs5.state_zipcode(vars, state_obj.fips, Census.ALL)
            all_data.extend(res)
        except:
            continue

    df = pd.DataFrame(all_data)
    if df.empty:
        return None

    df["population"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    df["income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    p_num = pd.to_numeric(df["B17001_002E"], errors="coerce")
    p_den = pd.to_numeric(df["B17001_001E"], errors="coerce")
    df["poverty_rate"] = (p_num / p_den.replace(0, 1)) * 100
    return df[df["income"] > 0]

def process_trends(files):
    if not files:
        return ""
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([(page.extract_text() or "") for page in reader.pages[:3]])
        except:
            pass
    return text[:15000]

# =============================================================================
# 4B) EVIDENCE HELPERS + LIGHT TABLE RENDERING
# =============================================================================
def _clean_str(x, max_len=260):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len]

def build_entity_evidence(df, entity, n=10):
    g = df[df["parent_company"] == entity].copy()
    if g.empty:
        return []
    g = g.sort_values("unique_scans_n", ascending=False)
    g = g.dropna(subset=["product_name"]).head(n)
    items = []
    for _, r in g.iterrows():
        items.append({
            "product_name": _clean_str(r.get("product_name"), 110),
            "claims_tags": _clean_str(r.get("labels_tags"), 180),
            "ingredients_snip": _clean_str(r.get("ingredients_text"), 260),
        })
    return items

def summarize_entity_signals(evidence_items):
    if not evidence_items:
        return "No evidence available."
    lines = []
    for it in evidence_items[:10]:
        lines.append(f"- {it['product_name']} | Claims: {it['claims_tags']} | Ing: {it['ingredients_snip']}")
    return "\n".join(lines)

def bullets_html(xs):
    if xs is None:
        return ""
    if isinstance(xs, list):
        return "<br>".join([f"• {re.sub(r'<','&lt;', str(x))}" for x in xs if str(x).strip()])
    return re.sub(r"<", "&lt;", str(xs))

def html_table(df: pd.DataFrame, max_rows: int = 200):
    """Force a light table (avoids st.dataframe dark-theme issues)."""
    if df is None or df.empty:
        st.markdown("<div class='small-muted'>No rows.</div>", unsafe_allow_html=True)
        return
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
        st.markdown(f"<div class='small-muted'>Showing first {max_rows} rows.</div>", unsafe_allow_html=True)
    st.write(df2.to_html(index=False, escape=False), unsafe_allow_html=True)

def _truncate(s, n=110):
    s = str(s).strip()
    return s if len(s) <= n else (s[: n - 1].rstrip() + "…")

def tile(title, subtitle, bullets, view_key, panel_name):
    bullets = [b for b in bullets if str(b).strip()][:3]  # less is more
    bullets_html_str = "".join([f"<div>• {_truncate(b, 140)}</div>" for b in bullets]) or "<div class='small-muted'>No content</div>"
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-sub">{subtitle}</div>
      <div class="card-bullets">{bullets_html_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tighter controls (buttons feel attached to tile)
    st.markdown('<div class="tight-controls"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("View details", key=f"{view_key}_view"):
            st.session_state.active_panel = panel_name
    with c2:
        if st.button("Hide", key=f"{view_key}_hide"):
            if st.session_state.active_panel == panel_name:
                st.session_state.active_panel = None

# =============================================================================
# 5) SIDEBAR (LOCK AFTER DIRECTIVE)
# =============================================================================
with st.sidebar:
    st.markdown("## Product Intelligence Agent")
    st.markdown("<div class='small-muted'>Run scan → pick focus → generate → one-page readout.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    GEMINI_API = None
    CENSUS_API = None
    uploaded_files = None
    execute = False

    if not st.session_state.ui_locked:
        GEMINI_API = st.text_input("Gemini API Key", type="password")
        CENSUS_API = st.text_input("Census API Key", type="password")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.session_state.sel_category = st.selectbox("Category", list(CATEGORY_MAP.keys()),
                                                     index=list(CATEGORY_MAP.keys()).index(st.session_state.sel_category)
                                                     if st.session_state.sel_category in CATEGORY_MAP else 0)
        st.session_state.sel_region = st.selectbox("Region", list(REGION_MAP.keys()),
                                                   index=list(REGION_MAP.keys()).index(st.session_state.sel_region)
                                                   if st.session_state.sel_region in REGION_MAP else 0)

        uploaded_files = st.file_uploader("Trend PDFs (optional)", type=["pdf"], accept_multiple_files=True)
        execute = st.button("Run scan", type="primary")

        st.markdown("<div class='small-muted'>Cleaning: gemini-3-flash-preview • Analysis: gemini-2.5-pro</div>", unsafe_allow_html=True)
    else:
        st.markdown("**Selections**")
        st.write(f"Category: **{st.session_state.sel_category}**")
        st.write(f"Region: **{st.session_state.sel_region}**")
        if st.session_state.sel_focus:
            st.write(f"Focus: **{st.session_state.sel_focus}**")
        if st.session_state.sel_competitors:
            st.write(f"Peers: **{', '.join(st.session_state.sel_competitors[:5])}**")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        if st.button("Edit selections"):
            st.session_state.ui_locked = False
            st.session_state.active_panel = None
            st.session_state.directive_result = None
            st.session_state.show_focus_block = True
            st.session_state.show_top_entities = True

# =============================================================================
# 6) MAIN
# =============================================================================
st.markdown("# Product Intelligence Hub")
st.markdown("<div class='small-muted'>One-page tiles. Click a tile to open the details drawer.</div>", unsafe_allow_html=True)

# Run scan
if execute:
    if not GEMINI_API or not CENSUS_API:
        st.error("Please provide both Gemini and Census API keys.")
        st.stop()

    with st.status("Working…", expanded=True) as status:
        st.write("Fetching demographics…")
        st.session_state.demographics_df = fetch_demographics(CENSUS_API, st.session_state.sel_region)

        st.write("Fetching market data…")
        st.session_state.market_df = fetch_market_intelligence(st.session_state.sel_category, GEMINI_API)

        st.write("Ingesting trend PDFs…")
        st.session_state.trends_text = process_trends(uploaded_files)

        st.session_state.data_fetched = True
        status.update(label="Data ready", state="complete")

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df

    if m_df is None or m_df.empty:
        st.error("No market data returned.")
        st.stop()
    if d_df is None or d_df.empty:
        st.error("No census data returned.")
        st.stop()

    # KPIs (keep subtle)
    k1, k2, k3, k4 = st.columns(4)
    parent_list = sorted(m_df["parent_company"].dropna().unique().tolist())
    k1.metric("SKUs", len(m_df))
    k2.metric("Entities", len(parent_list))
    k3.metric("Avg income", f"${d_df['income'].mean():,.0f}")
    k4.metric("Poverty", f"{d_df['poverty_rate'].mean():.1f}%")

    st.markdown("<div class='section-label'>Market snapshot</div>", unsafe_allow_html=True)

    # Focus selection block (hide after generate)
    if st.session_state.show_focus_block and not st.session_state.ui_locked:
        c1, c2 = st.columns([2, 1])
        with c1:
            my_brand = st.selectbox(
                "Focus entity",
                parent_list,
                index=0 if (st.session_state.sel_focus is None or st.session_state.sel_focus not in parent_list)
                else parent_list.index(st.session_state.sel_focus),
            )
            st.session_state.sel_focus = my_brand
            # collapse top entities after focus chosen
            st.session_state.show_top_entities = False
        with c2:
            st.markdown("<div class='small-muted' style='padding-top: 1.65rem;'>Pick the focal entity. Then generate insights.</div>", unsafe_allow_html=True)

        # Optional expander (defaults collapsed after focus)
        with st.expander("Top entities (SKU count)", expanded=st.session_state.show_top_entities):
            st.bar_chart(m_df["parent_company"].value_counts().head(10))
    else:
        my_brand = st.session_state.sel_focus or (parent_list[0] if parent_list else None)

    comp_list = (
        m_df[m_df["parent_company"] != my_brand]["parent_company"]
        .value_counts().head(6).index.tolist()
        if my_brand else []
    )

    st.markdown("<div class='section-label'>Directive</div>", unsafe_allow_html=True)

    # Generate row (compact)
    c_gen1, c_gen2 = st.columns([1, 3])
    with c_gen1:
        generate = st.button("Generate", type="primary", disabled=st.session_state.ui_locked)
    with c_gen2:
        st.markdown("<div class='small-muted'>Generates second-order insights (evidence → inference → implication). Includes claim feasibility checks.</div>", unsafe_allow_html=True)

    result = st.session_state.directive_result

    if generate:
        st.session_state.sel_focus = my_brand
        st.session_state.sel_competitors = comp_list
        st.session_state.ui_locked = True
        st.session_state.show_focus_block = False

        genai.configure(api_key=GEMINI_API)
        model = genai.GenerativeModel("gemini-2.5-pro")

        my_evidence_txt = summarize_entity_signals(build_entity_evidence(m_df, my_brand, n=10))
        comp_evidence_txt = "\n\n".join(
            [f"{c}:\n{summarize_entity_signals(build_entity_evidence(m_df, c, n=8))}" for c in comp_list]
        )

        total_skus = len(m_df)
        my_skus = len(m_df[m_df["parent_company"] == my_brand])
        trends = st.session_state.trends_text or ""

        prompt = f"""
ACT AS: Product Intelligence Lead for a CPG portfolio.
GOAL: Produce insights that are "second-order": not obvious observations, but evidence-led inferences and implications.

HARD RULES:
1) Each insight must be structured as:
   - "observation": what the dataset shows (1 sentence)
   - "evidence": 2–3 proof points from the SKU snippets (explicit examples like product names, claims tags, ingredient snippets)
   - "inference": what it likely means (1 sentence, causal language)
   - "implication": so what / what decision it changes (1 sentence)
2) Do NOT state trivialities ("we all sell peanuts"). If the observation is obvious, elevate it to a second-order implication.
3) Claims recommendations MUST be feasible:
   - If ingredients/labels indicate a claim is blocked (e.g., gelatin conflicts with vegan), mark it "blocked" and say why.
   - Provide "what would need to change" to unlock it (reformulation, label substantiation, certification, etc.).
4) If evidence is weak, say so and specify the missing data (POS, retailer item file, claims validation, etc.).

CONTEXT:
- Category: {st.session_state.sel_category}
- Region: {st.session_state.sel_region}
- Dataset size: {total_skus} SKUs
- Focus entity: {my_brand} ({my_skus} observed SKUs)
- Demographics: Avg Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%

EVIDENCE — FOCUS ENTITY SKU SNIPPETS:
{my_evidence_txt}

EVIDENCE — TOP COMPETITORS SKU SNIPPETS:
{comp_evidence_txt}

TRENDS (PDF snippets; may be empty):
{trends}

RETURN JSON ONLY, EXACT SCHEMA:

{{
  "executive_summary": {{
    "bluf": "2 sentences max",
    "key_insights": [
      {{
        "observation": "…",
        "evidence": ["…","…"],
        "inference": "…",
        "implication": "…"
      }}
    ],
    "gaps_and_risks": ["…","…"]
  }},
  "market_structure": {{
    "branded_vs_private_label": [
      {{
        "observation": "…",
        "evidence": ["…","…"],
        "inference": "…",
        "implication": "…"
      }}
    ],
    "competitive_roles": [
      {{
        "entity": "name",
        "role": "incumbent | premium | value | retailer_brand | niche",
        "proof_points": ["…","…","…"]
      }}
    ]
  }},
  "occasion_cards": [
    {{
      "occasion_name": "short title",
      "slide_headline": "short headline",
      "definition": "1 sentence",
      "who_wins_today": "entity/private label",
      "winning_offer": ["3 bullets"],
      "gap_for_{my_brand}": ["3 bullets"],
      "moves_for_{my_brand}": ["5 bullets (must be specific)"]
    }}
  ],
  "claims_strategy": {{
    "category_claim_patterns": [
      {{
        "pattern": "e.g., gluten-free as mainstream trust cue",
        "evidence": ["…","…"],
        "inference": "…",
        "implication": "…"
      }}
    ],
    "opportunity_claims_for_{my_brand}": [
      {{
        "claim": "…",
        "status": "feasible | blocked | unclear",
        "why": "1 sentence",
        "evidence_or_conflict": ["…","…"],
        "what_would_need_to_change": ["…","…"],
        "when_to_use": "occasion / segment"
      }}
    ]
  }},
  "ingredient_audit": [
    {{
      "ingredient_type": "e.g., sweetener | oil | preservative | flavor system | allergen cue",
      "insight": {{
        "observation": "…",
        "evidence": ["…","…"],
        "inference": "…",
        "implication": "…"
      }},
      "{my_brand}_examples": ["…","…","…"],
      "competitor_examples": [
        {{"entity":"…","examples":["…","…","…"]}}
      ]
    }}
  ],
  "strategic_questions": ["…","…","…"]
}}
"""
        try:
            with st.spinner("Generating…"):
                response = model.generate_content(prompt)
            res_txt = re.sub(r"```json\s?|```", "", response.text).strip()
            result = json.loads(res_txt)
            st.session_state.directive_result = result
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    # ==========================
    # One-page tiles
    # ==========================
    if result:
        es = result.get("executive_summary", {})
        ms = result.get("market_structure", {})
        bpl = ms.get("branded_vs_private_label", [])
        occ = result.get("occasion_cards", [])
        cs = result.get("claims_strategy", {})
        ing = result.get("ingredient_audit", [])

        # Less text: pull just 2-3 “implications” style snippets for tiles
        def _tile_from_insights(insights, take=3):
            out = []
            for it in (insights or [])[:take]:
                # prefer implication (second-order)
                imp = it.get("implication", "")
                inf = it.get("inference", "")
                if imp:
                    out.append(imp)
                elif inf:
                    out.append(inf)
                else:
                    out.append(it.get("observation", ""))
            return out[:take]

        tile_exec = _tile_from_insights(es.get("key_insights", []), take=3)
        tile_market = _tile_from_insights(bpl, take=3)

        tile_occ = []
        for c in occ[:3]:
            tile_occ.append(f"{c.get('occasion_name','Occasion')}: {c.get('slide_headline','')}".strip().strip(":"))
        tile_occ = tile_occ[:3]

        patterns = cs.get("category_claim_patterns", [])
        tile_claims = _tile_from_insights(patterns, take=3)

        tile_ing = []
        for row in ing[:3]:
            insight = (row.get("insight") or {})
            tile_ing.append(insight.get("implication") or insight.get("inference") or insight.get("observation") or "")
        tile_ing = [t for t in tile_ing if t][:3]

        st.markdown("<div class='section-label'>One-page readout</div>", unsafe_allow_html=True)

        r0c1, r0c2 = st.columns(2)
        with r0c1:
            tile("Executive summary", "Second-order insights (so what)", tile_exec, "exec", "exec")
        with r0c2:
            tile("Market structure", "Branded vs private label dynamics", tile_market, "market", "market")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            tile("Occasions", "Where value concentrates", tile_occ, "occ", "occasions")
        with r1c2:
            tile("Claims strategy", "Feasible + defensible plays", tile_claims, "claims", "claims")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            tile("Ingredient audit", "Drivers of perceived quality / cost", tile_ing, "ing", "ingredients")
        with r2c2:
            # a lightweight tile to open mapping audit (no black dataframe)
            tile("Entity normalization", "Validate mappings before presenting", ["Spot-check surprising parents (e.g., PepsiCo in nuts) before using counts."], "map", "mapping")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ==========================
        # Details drawer
        # ==========================
        panel = st.session_state.active_panel
        if not panel:
            st.markdown("<div class='small-muted'>Click any tile to open detail here.</div>", unsafe_allow_html=True)

        if panel:
            st.markdown("## Details")
            st.markdown("<div class='small-muted'>Observation → Evidence → Inference → Implication</div>", unsafe_allow_html=True)

            if panel == "exec":
                st.markdown("### Executive summary")
                st.info(es.get("bluf", ""))
                for i, it in enumerate(es.get("key_insights", []), start=1):
                    with st.expander(f"Insight {i}", expanded=(i == 1)):
                        st.markdown(f"**Observation:** {it.get('observation','')}")
                        st.markdown("**Evidence:**")
                        for ev in it.get("evidence", []):
                            st.write(f"• {ev}")
                        st.markdown(f"**Inference:** {it.get('inference','')}")
                        st.markdown(f"**Implication:** {it.get('implication','')}")
                st.markdown("**Gaps & risks**")
                for g in es.get("gaps_and_risks", []):
                    st.write(f"• {g}")

            elif panel == "market":
                st.markdown("### Market structure")
                for i, it in enumerate(bpl, start=1):
                    with st.expander(f"Dynamic {i}", expanded=(i == 1)):
                        st.markdown(f"**Observation:** {it.get('observation','')}")
                        st.markdown("**Evidence:**")
                        for ev in it.get("evidence", []):
                            st.write(f"• {ev}")
                        st.markdown(f"**Inference:** {it.get('inference','')}")
                        st.markdown(f"**Implication:** {it.get('implication','')}")
                roles = ms.get("competitive_roles", [])
                if roles:
                    st.markdown("### Competitive roles")
                    html_table(pd.DataFrame(roles))

            elif panel == "occasions":
                st.markdown("### Occasion cards")
                for c in occ:
                    with st.expander(f"{c.get('occasion_name','Occasion')} — {c.get('slide_headline','')}", expanded=False):
                        st.markdown(f"**Definition:** {c.get('definition','')}")
                        st.markdown(f"**Who wins today:** {c.get('who_wins_today','')}")
                        st.markdown("**Winning offer**")
                        for b in c.get("winning_offer", []):
                            st.write(f"• {b}")
                        st.markdown(f"**Gap for {my_brand}**")
                        for b in c.get(f"gap_for_{my_brand}", []):
                            st.write(f"• {b}")
                        st.markdown(f"**Moves for {my_brand}**")
                        for b in c.get(f"moves_for_{my_brand}", []):
                            st.write(f"• {b}")

            elif panel == "claims":
                st.markdown("### Claims strategy (feasibility-checked)")
                st.markdown("#### Category claim patterns")
                for i, p in enumerate(cs.get("category_claim_patterns", []), start=1):
                    with st.expander(f"Pattern {i}: {p.get('pattern','')}", expanded=(i == 1)):
                        st.markdown(f"**Evidence:**")
                        for ev in p.get("evidence", []):
                            st.write(f"• {ev}")
                        st.markdown(f"**Inference:** {p.get('inference','')}")
                        st.markdown(f"**Implication:** {p.get('implication','')}")

                st.markdown("#### Opportunity claims (feasible vs blocked)")
                opp = cs.get(f"opportunity_claims_for_{my_brand}", [])
                if opp:
                    opp_df = pd.DataFrame(opp)
                    html_table(opp_df)

            elif panel == "ingredients":
                st.markdown("### Ingredient audit")
                for i, row in enumerate(ing, start=1):
                    insight = row.get("insight") or {}
                    with st.expander(f"{i}. {row.get('ingredient_type','Ingredient')}", expanded=(i == 1)):
                        st.markdown(f"**Observation:** {insight.get('observation','')}")
                        st.markdown("**Evidence:**")
                        for ev in insight.get("evidence", []):
                            st.write(f"• {ev}")
                        st.markdown(f"**Inference:** {insight.get('inference','')}")
                        st.markdown(f"**Implication:** {insight.get('implication','')}")
                        st.markdown(f"**{my_brand} examples**")
                        for ex in row.get(f"{my_brand}_examples", []):
                            st.write(f"• {ex}")
                        st.markdown("**Competitor examples**")
                        for ce in row.get("competitor_examples", []):
                            st.write(f"**{ce.get('entity','')}**")
                            for ex in ce.get("examples", []):
                                st.write(f"• {ex}")

            elif panel == "mapping":
                st.markdown("### Entity normalization audit")
                st.markdown("<div class='small-muted'>Spot-check out-of-place parents before presenting entity counts.</div>", unsafe_allow_html=True)
                audit_df = m_df[["brands", "parent_company"]].drop_duplicates().sort_values(["parent_company", "brands"])
                html_table(audit_df, max_rows=250)

            c_close1, c_close2, _ = st.columns([1, 1, 3])
            with c_close1:
                if st.button("Close details"):
                    st.session_state.active_panel = None
            with c_close2:
                if st.button("Reset view"):
                    st.session_state.active_panel = None

else:
    st.markdown("<div class='small-muted'>Run a scan to begin.</div>", unsafe_allow_html=True)

