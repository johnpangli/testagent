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
# 1) PAGE CONFIG + "GOOGLE/APPLE" CLEAN UI
# =============================================================================
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="◼", layout="wide")

st.markdown("""
<style>
/* ---------- Layout / Typography ---------- */
div.block-container { padding-top: 1.25rem; max-width: 1400px; }
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; }

h1, h2, h3, h4 { letter-spacing: -0.02em; }
h1 { font-size: 28px !important; margin-bottom: 0.25rem; }
h2 { font-size: 18px !important; margin-top: 0.75rem; margin-bottom: 0.25rem; }
h3 { font-size: 15px !important; margin-top: 0.75rem; margin-bottom: 0.25rem; }

.small-muted { color: #64748b; font-size: 12px; line-height: 1.35; }
.kpi-note { color: #64748b; font-size: 11px; }

/* ---------- Buttons ---------- */
.stButton>button {
  width: 100%;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.55rem 0.9rem;
  border: 1px solid #e2e8f0;
}
.stButton>button:hover { border-color: #cbd5e1; }

/* ---------- Section header ---------- */
.section-header {
  background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
  color: white;
  padding: 14px 18px;
  border-radius: 14px;
  margin: 18px 0 14px 0;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

/* ---------- Tiles (Quadrants) ---------- */
.tile {
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  background: #ffffff;
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.04);
  min-height: 230px;
}
.tile-title { font-size: 14px; font-weight: 750; color: #0f172a; margin-bottom: 4px; }
.tile-sub { font-size: 12px; color: #64748b; margin-bottom: 10px; }
.tile-bullets { font-size: 13px; color: #0f172a; line-height: 1.35; }
.tile-bullets div { margin: 7px 0; }
.tile-footer { margin-top: 12px; display: flex; gap: 8px; }

/* ---------- Tables (clean) ---------- */
table { width: 100%; border-collapse: collapse; }
th {
  background-color: #f8fafc;
  color: #0f172a !important;
  text-align: left;
  padding: 10px;
  font-weight: 700;
  border-bottom: 1px solid #e2e8f0;
}
td {
  padding: 9px 10px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: top;
}

/* ---------- Expander styling (subtle) ---------- */
details summary {
  font-size: 13px;
  font-weight: 650;
  color: #0f172a;
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

# UI/Flow state
if "active_panel" not in st.session_state:
    st.session_state.active_panel = None  # "market" | "occasions" | "claims" | "ingredients"
if "directive_result" not in st.session_state:
    st.session_state.directive_result = None
if "ui_locked" not in st.session_state:
    st.session_state.ui_locked = False

# Persist selections after lock
if "sel_region" not in st.session_state:
    st.session_state.sel_region = "Midwest"
if "sel_category" not in st.session_state:
    st.session_state.sel_category = "Snack Nuts"
if "sel_focus" not in st.session_state:
    st.session_state.sel_focus = None
if "sel_competitors" not in st.session_state:
    st.session_state.sel_competitors = []

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
1) CONSOLIDATE VARIATIONS: "Blue Diamond", "Blue Diamond Almonds", "Blue Diamond Growers" -> "Blue Diamond Growers".
2) RESOLVE PARENTS: "Wright", "Wright Brand", "Wright Foods" -> "Tyson Foods".
3) RETAILER BRANDS: "365", "Whole Foods", "365 Everyday Value" -> "Amazon/Whole Foods".
4) PRIVATE LABEL: "Great Value" -> "Walmart", "Kirkland" -> "Costco".
5) HIERARCHY: Always aim for the ultimate corporate owner.

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
    headers = {"User-Agent": "StrategicIntelligenceHub/3.0"}
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

    # Entity resolution (MDM) — Gemini-3-Flash-Preview
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
# 4B) EVIDENCE PACK HELPERS + READABLE BULLETS
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
            "product_name": _clean_str(r.get("product_name"), 100),
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

# =============================================================================
# UI HELPERS: Tiles + Drawer
# =============================================================================
def tile_block(title, subtitle, bullets):
    bullets_html_str = "".join([f"<div>• {b}</div>" for b in bullets if str(b).strip()])
    if not bullets_html_str:
        bullets_html_str = "<div class='small-muted'>No content</div>"
    st.markdown(f"""
    <div class="tile">
      <div class="tile-title">{title}</div>
      <div class="tile-sub">{subtitle}</div>
      <div class="tile-bullets">{bullets_html_str}</div>
    </div>
    """, unsafe_allow_html=True)

def tile_controls(view_key, panel_name):
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("View details", key=f"{view_key}_view"):
            st.session_state.active_panel = panel_name
    with c2:
        if st.button("Collapse", key=f"{view_key}_collapse"):
            if st.session_state.active_panel == panel_name:
                st.session_state.active_panel = None

# =============================================================================
# 5) SIDEBAR (LOCK AFTER DIRECTIVE)
# =============================================================================
with st.sidebar:
    st.markdown("## Strategy Agent")
    st.markdown("<div class='small-muted'>One-page readout. Drill into tiles only when needed.</div>", unsafe_allow_html=True)
    st.divider()

    if not st.session_state.ui_locked:
        GEMINI_API = st.text_input("Gemini API Key", type="password")
        CENSUS_API = st.text_input("Census API Key", type="password")

        st.divider()
        st.session_state.sel_region = st.selectbox("Region", list(REGION_MAP.keys()), index=list(REGION_MAP.keys()).index(st.session_state.sel_region) if st.session_state.sel_region in REGION_MAP else 0)
        st.session_state.sel_category = st.selectbox("Category", list(CATEGORY_MAP.keys()), index=list(CATEGORY_MAP.keys()).index(st.session_state.sel_category) if st.session_state.sel_category in CATEGORY_MAP else 0)

        uploaded_files = st.file_uploader("Trend PDFs (optional)", type=["pdf"], accept_multiple_files=True)

        execute = st.button("Run market scan", type="primary")
        st.markdown("<div class='small-muted'>Cleaning: Gemini-3-Flash-Preview · Strategy: Gemini-2.5-Pro</div>", unsafe_allow_html=True)
    else:
        # Locked view: show compact selections
        st.markdown("**Selections**")
        st.write(f"Category: **{st.session_state.sel_category}**")
        st.write(f"Region: **{st.session_state.sel_region}**")

        if st.session_state.sel_focus:
            st.write(f"Focus: **{st.session_state.sel_focus}**")
        if st.session_state.sel_competitors:
            st.write(f"Peers: **{', '.join(st.session_state.sel_competitors[:5])}**")

        st.divider()
        if st.button("Edit selections"):
            st.session_state.ui_locked = False
            st.session_state.active_panel = None
            st.session_state.directive_result = None

        execute = False
        GEMINI_API = None
        CENSUS_API = None
        uploaded_files = None

# =============================================================================
# 6) MAIN
# =============================================================================
st.markdown("# Strategic Intelligence Hub")
st.markdown("<div class='small-muted'>Clean tiles. Consulting-style proof points. Click any tile to drill down.</div>", unsafe_allow_html=True)

# Run scan
if "execute" not in locals():
    execute = False

if execute:
    if not GEMINI_API:
        st.error("Please provide a Gemini API key.")
        st.stop()
    if not CENSUS_API:
        st.error("Please provide a Census API key.")
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

# If we already have data
if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df

    if m_df is None or m_df.empty:
        st.error("No market data returned.")
        st.stop()
    if d_df is None or d_df.empty:
        st.error("No census data returned.")
        st.stop()

    # KPIs row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs", len(m_df))
    parent_list = sorted(m_df["parent_company"].dropna().unique().tolist())
    k2.metric("Entities", len(parent_list))
    k3.metric("Avg income", f"${d_df['income'].mean():,.0f}")
    k4.metric("Poverty", f"{d_df['poverty_rate'].mean():.1f}%")

    st.markdown("<div class='section-header'>Market snapshot</div>", unsafe_allow_html=True)

    # Focus selection (kept in main content pre-lock)
    sel_cols = st.columns([2, 1])
    with sel_cols[0]:
        my_brand = st.selectbox("Focus entity", parent_list, index=0 if st.session_state.sel_focus is None or st.session_state.sel_focus not in parent_list else parent_list.index(st.session_state.sel_focus))
    with sel_cols[1]:
        st.markdown("<div class='small-muted' style='padding-top: 1.6rem;'>Choose the focal entity, then generate a directive.</div>", unsafe_allow_html=True)

    # Simple competitor list (by presence)
    comp_list = (
        m_df[m_df["parent_company"] != my_brand]["parent_company"]
        .value_counts()
        .head(6)
        .index.tolist()
    )

    # Top bar chart (compact)
    with st.expander("Top entities (SKU count)", expanded=False):
        st.bar_chart(m_df["parent_company"].value_counts().head(10))

    # Generate directive control (once we generate, lock the sidebar + store selections)
    st.markdown("<div class='section-header'>Strategy directive</div>", unsafe_allow_html=True)
    gen_cols = st.columns([1.2, 1.2, 1.6])
    with gen_cols[0]:
        generate = st.button("Generate directive", type="primary")
    with gen_cols[1]:
        st.markdown("<div class='small-muted'>Uses Gemini-2.5-Pro with SKU-level evidence.</div>", unsafe_allow_html=True)
    with gen_cols[2]:
        st.markdown("<div class='small-muted'>Output is directional without POS. Treat as hypothesis generator.</div>", unsafe_allow_html=True)

    # If directive already exists, use it
    result = st.session_state.directive_result

    if generate:
        # Lock selections after generation
        st.session_state.sel_focus = my_brand
        st.session_state.sel_competitors = comp_list
        st.session_state.ui_locked = True

        # Strategy model: Gemini-2.5-Pro
        genai.configure(api_key=GEMINI_API)
        model = genai.GenerativeModel("gemini-2.5-pro")

        my_evidence_txt = summarize_entity_signals(build_entity_evidence(m_df, my_brand, n=10))
        comp_evidence_txt = "\n\n".join(
            [f"{c}:\n{summarize_entity_signals(build_entity_evidence(m_df, c, n=8))}" for c in comp_list]
        )

        total_skus = len(m_df)
        my_skus = len(m_df[m_df["parent_company"] == my_brand])

        prompt = f"""
ACT AS: Chief Strategy Officer for a CPG firm.
GOAL: Produce strategy that can expand into slides (clear headers, MECE structure, specific proof points).
CONSTRAINTS:
- Use only the evidence provided (OpenFoodFacts SKU snippets + uploaded trend PDFs + Census summary).
- Avoid generic CPG fluff. If evidence is weak, say so and propose what data you’d request next (POS, retailer, etc.).
- Keep each text field max 2 sentences unless it's a list. Prefer lists over paragraphs.

CONTEXT:
- Category: {st.session_state.sel_category}
- Region: {st.session_state.sel_region}
- Category dataset size (SKUs pulled): {total_skus}
- Focus entity: {my_brand} (SKUs observed: {my_skus})
- Demographics: Avg Income ${d_df['income'].mean():,.0f}, Avg Poverty {d_df['poverty_rate'].mean():.1f}%

EVIDENCE — MY ENTITY (SKU / Claims / Ingredients):
{my_evidence_txt}

EVIDENCE — TOP COMPETITORS (SKU / Claims / Ingredients):
{comp_evidence_txt}

TRENDS (from PDFs; may be empty):
{st.session_state.trends_text}

TASK: Return JSON ONLY (no markdown) with this EXACT schema:

{{
  "executive_summary": {{
    "bluf": "2 sentences max, crisp.",
    "what_we_know": ["3 bullets grounded in evidence"],
    "what_we_dont_know": ["2 bullets: data gaps / risks"]
  }},
  "market_structure": {{
    "branded_vs_private_label": {{
      "observations": ["3 bullets: how private label vs branded appears in THIS dataset"],
      "implications": ["3 bullets: what it means strategically"],
      "watchouts": ["2 bullets: data limitations or possible misclassification"]
    }},
    "competitive_arena": [
      {{
        "entity": "name",
        "role": "incumbent | premium | value | retailer_brand | niche",
        "proof_points": ["3 bullets referencing evidence patterns (claims/ingredients/product types)"]
      }}
    ]
  }},
  "occasion_cards": [
    {{
      "occasion_name": "short title",
      "definition": "1 sentence: what the shopper is doing / why",
      "who_wins_today": "competitor or private label",
      "winning_offer": ["3 bullets: claims, ingredients, formats, pack cues inferred from evidence"],
      "gap_for_{my_brand}": ["3 bullets: what we're missing"],
      "moves_for_{my_brand}": ["5 bullets: specific actions (claims, ingredients, pack architecture, price/pack, channel)"],
      "slide_headline": "one slide headline you’d put on a deck"
    }},
    {{
      "occasion_name": "...",
      "definition": "...",
      "who_wins_today": "...",
      "winning_offer": ["..."],
      "gap_for_{my_brand}": ["..."],
      "moves_for_{my_brand}": ["..."],
      "slide_headline": "..."
    }},
    {{
      "occasion_name": "...",
      "definition": "...",
      "who_wins_today": "...",
      "winning_offer": ["..."],
      "gap_for_{my_brand}": ["..."],
      "moves_for_{my_brand}": ["..."],
      "slide_headline": "..."
    }}
  ],
  "claims_strategy": {{
    "category_claims_that_win": ["5 bullets"],
    "competitor_claim_patterns": [
      {{
        "entity": "name",
        "claims": ["3 bullets"],
        "proof_points": ["2 bullets referencing evidence"]
      }}
    ],
    "opportunity_claims_for_{my_brand}": ["5 bullets"]
  }},
  "ingredient_audit": [
    {{
      "ingredient_type": "e.g., sweetener | oil | preservative | protein source | flavor system | allergen cue",
      "{my_brand}": ["3 bullets of observed examples"],
      "competitor_1": {{
        "entity": "name",
        "examples": ["3 bullets"]
      }},
      "competitor_2": {{
        "entity": "name",
        "examples": ["3 bullets"]
      }},
      "implication": "1 sentence max"
    }}
  ],
  "strategic_questions": [
    "3 hard questions (pack architecture / assortment / price-pack / channel)"
  ]
}}

IMPORTANT:
- Use real competitor names from the competitor list when possible.
- If you can’t find evidence for a claim/ingredient, label it as 'Not observed in evidence' rather than inventing.
"""

        try:
            with st.spinner("Generating strategy…"):
                response = model.generate_content(prompt)
            res_txt = re.sub(r"```json\s?|```", "", response.text).strip()
            result = json.loads(res_txt)
            st.session_state.directive_result = result
        except Exception as e:
            st.error(f"Strategy generation failed: {e}")
            st.stop()

    # ----------------------------
    # One-page tiles (Quadrants)
    # ----------------------------
    if result:
        es = result.get("executive_summary", {})
        ms = result.get("market_structure", {})
        bpl = ms.get("branded_vs_private_label", {})
        occ = result.get("occasion_cards", [])
        cs = result.get("claims_strategy", {})
        ing = result.get("ingredient_audit", [])

        tile_market = (bpl.get("observations", [])[:2] + bpl.get("implications", [])[:2] + bpl.get("watchouts", [])[:1])
        tile_occ = []
        for c in occ[:3]:
            nm = c.get("occasion_name", "Occasion")
            hd = c.get("slide_headline", "")
            tile_occ.append(f"{nm}: {hd}".strip().strip(":"))
        tile_occ = tile_occ[:5]

        tile_claims = (cs.get("category_claims_that_win", [])[:3] + cs.get(f"opportunity_claims_for_{my_brand}", [])[:2])

        tile_ing = []
        for row in ing[:5]:
            tile_ing.append(f"{row.get('ingredient_type','Ingredient')}: {row.get('implication','')}")
        tile_ing = tile_ing[:5]

        st.markdown("<div class='section-header'>One-page strategy readout</div>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            tile_block("Market Structure", "Branded vs private label + competitive roles", tile_market)
            tile_controls("tile_market", "market")
        with r1c2:
            tile_block("Occasions", "Three MECE occasions with slide headlines", tile_occ)
            tile_controls("tile_occ", "occasions")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            tile_block("Claims Strategy", "Winning claims + opportunity claims", tile_claims)
            tile_controls("tile_claims", "claims")
        with r2c2:
            tile_block("Ingredient Audit", "Key differences + why they matter", tile_ing)
            tile_controls("tile_ing", "ingredients")

        # ----------------------------
        # Detail Drawer (single area)
        # ----------------------------
        st.divider()
        panel = st.session_state.active_panel

        if not panel:
            st.markdown("<div class='small-muted'>Tip: click “View details” on any tile to drill into evidence and tables.</div>", unsafe_allow_html=True)

        if panel:
            st.markdown("## Details")
            st.markdown("<div class='small-muted'>This drawer changes based on the tile you selected.</div>", unsafe_allow_html=True)

            if panel == "market":
                st.markdown("### Market Structure")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Observations**")
                    for b in bpl.get("observations", []):
                        st.write(f"• {b}")
                with c2:
                    st.markdown("**Implications**")
                    for b in bpl.get("implications", []):
                        st.write(f"• {b}")
                with c3:
                    st.markdown("**Watchouts**")
                    for b in bpl.get("watchouts", []):
                        st.write(f"• {b}")

                arena = ms.get("competitive_arena", [])
                if arena:
                    st.markdown("**Competitive arena**")
                    st.write(pd.DataFrame(arena).to_html(index=False), unsafe_allow_html=True)

                with st.expander("Entity mapping audit (raw brand → parent)", expanded=False):
                    st.dataframe(m_df[["brands", "parent_company"]].drop_duplicates())

            elif panel == "occasions":
                st.markdown("### Occasion Cards")
                for card in occ:
                    nm = card.get("occasion_name", "Occasion")
                    hd = card.get("slide_headline", "")
                    with st.expander(f"{nm} — {hd}", expanded=False):
                        st.markdown(f"**Definition:** {card.get('definition','')}")
                        st.markdown(f"**Who wins today:** {card.get('who_wins_today','')}")
                        st.markdown("**Winning offer**")
                        for b in card.get("winning_offer", []):
                            st.write(f"• {b}")
                        st.markdown(f"**Gap for {my_brand}**")
                        for b in card.get(f"gap_for_{my_brand}", []):
                            st.write(f"• {b}")
                        st.markdown(f"**Moves for {my_brand}**")
                        for b in card.get(f"moves_for_{my_brand}", []):
                            st.write(f"• {b}")

            elif panel == "claims":
                st.markdown("### Claims Strategy")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Category claims that win**")
                    for b in cs.get("category_claims_that_win", []):
                        st.write(f"• {b}")
                with c2:
                    st.markdown(f"**Opportunity claims for {my_brand}**")
                    for b in cs.get(f"opportunity_claims_for_{my_brand}", []):
                        st.write(f"• {b}")

                patterns = cs.get("competitor_claim_patterns", [])
                if patterns:
                    st.markdown("**Competitor patterns (with proof points)**")
                    st.write(pd.DataFrame(patterns).to_html(index=False), unsafe_allow_html=True)

            elif panel == "ingredients":
                st.markdown("### Ingredient Audit")
                ing_list = result.get("ingredient_audit", [])
                if ing_list:
                    ing_df = pd.DataFrame(ing_list)

                    if my_brand in ing_df.columns:
                        ing_df[my_brand] = ing_df[my_brand].apply(bullets_html)

                    def _comp_block(c):
                        if not isinstance(c, dict):
                            return ""
                        name = c.get("entity", "")
                        ex = c.get("examples", [])
                        return f"<b>{name}</b><br>{bullets_html(ex)}"

                    if "competitor_1" in ing_df.columns:
                        ing_df["competitor_1"] = ing_df["competitor_1"].apply(_comp_block)
                    if "competitor_2" in ing_df.columns:
                        ing_df["competitor_2"] = ing_df["competitor_2"].apply(_comp_block)

                    st.write(ing_df.to_html(index=False, escape=False), unsafe_allow_html=True)
                else:
                    st.markdown("<div class='small-muted'>No ingredient audit returned.</div>", unsafe_allow_html=True)

            st.divider()

    else:
        st.markdown("<div class='small-muted'>Generate a directive to populate the one-page tiles.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='small-muted'>Run a market scan to begin.</div>", unsafe_allow_html=True)
