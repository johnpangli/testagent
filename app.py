import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import re
import us
from concurrent.futures import ThreadPoolExecutor, as_completed
from census import Census
import google.generativeai as genai
from pypdf import PdfReader

# =============================================================================
# 1) UI CONFIGURATION & PROFESSIONAL STYLING
# =============================================================================
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    div.block-container {padding-top: 1.5rem; max-width: 1400px;}
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --text-dark: #1e293b;
    }
    .stButton>button { 
        width: 100%; border-radius: 8px; font-weight: 600;
        padding: 0.6rem 1rem; transition: all 0.2s;
    }
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-size: 18px; font-weight: 600;
    }
    /* TABLE STYLING FIX */
    table {width: 100%; border-collapse: collapse;}
    th {
        background-color: #f0f2f6; 
        color: #000000 !important;
        text-align: left;
        padding: 10px;
        font-weight: bold;
    }
    td { padding: 8px; border-bottom: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 2) SESSION STATE MANAGEMENT
# =============================================================================
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'market_df' not in st.session_state:
    st.session_state.market_df = None
if 'demographics_df' not in st.session_state:
    st.session_state.demographics_df = None
if 'trends_text' not in st.session_state:
    st.session_state.trends_text = ""

# =============================================================================
# 3) THE "ULTIMATE PARENT" MDM LOGIC (LOCKED TO GEMINI-3-FLASH-PREVIEW)
# =============================================================================
def get_canonical_parent_map(messy_brands, api_key):
    """
    Consolidates messy brand strings into Ultimate Corporate Parents.
    Model locked to: gemini-3-flash-preview
    """
    if not messy_brands or not api_key:
        return {}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')  # MANDATORY MODEL LOCK

    prompt = f"""
ACT AS: Enterprise Master Data Management (MDM) Specialist for a CPG Firm.
TASK: Clean this list of messy brand strings and map them to their ONE true Parent Company.

LOGIC RULES:
1. CONSOLIDATE VARIATIONS: "Blue Diamond", "Blue Diamond Almonds", "Blue Diamond Growers" -> "Blue Diamond Growers".
2. RESOLVE PARENTS: "Wright", "Wright Brand", "Wright Foods" -> "Tyson Foods".
3. RETAILER BRANDS: "365", "Whole Foods", "365 Everyday Value" -> "Amazon/Whole Foods".
4. PRIVATE LABEL: "Great Value" -> "Walmart", "Kirkland" -> "Costco".
5. HIERARCHY: Always aim for the ultimate corporate owner (e.g., Hormel, Kraft Heinz, General Mills).

LIST TO RESOLVE:
{messy_brands}

RETURN ONLY VALID JSON OBJECT (No markdown, no text):
{{
  "Mapping": [
    {{"raw": "Messy Name", "canonical_parent": "Clean Parent Company"}},
    ...
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json\s?|```', '', response.text).strip()
        data = json.loads(clean_json)
        return {item['raw']: item['canonical_parent'] for item in data.get('Mapping', [])}
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
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]
}

CATEGORY_MAP = {
    "Bacon": "bacons",
    "Peanut Butter": "peanut-butters",
    "Snack Nuts": "nuts",
    "Beef Jerky": "meat-snacks",
    "Coffee": "coffees",
    "Cereal": "breakfast-cereals",
    "Chips": "chips"
}

def fetch_market_intelligence(category, api_key):
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    headers = {'User-Agent': 'StrategicIntelligenceHub/2.0'}
    all_products = []

    status_text = st.empty()

    for page in range(1, 6):
        status_text.text(f"🚜 Scouting Page {page} via Category Tag...")
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
            products = r.json().get('products', [])
            if not products:
                break
            all_products.extend(products)
            time.sleep(0.5)
        except:
            break

    status_text.empty()
    df = pd.DataFrame(all_products)
    if df.empty:
        return df

    df['brands'] = df['brands'].astype(str).str.strip().str.strip(',')
    df = df[~df['brands'].isin(['nan', 'None', '', 'Unknown', 'null'])]
    df = df.drop_duplicates(subset=['product_name'])

    # Run the "Ultimate Parent" Cleaner (Gemini-3-Flash-Preview)
    unique_messy = df['brands'].unique().tolist()
    with st.spinner(f"AI Entity Resolution: Consolidating {len(unique_messy)} brands..."):
        parent_map = get_canonical_parent_map(unique_messy, api_key)

    df['parent_company'] = df['brands'].map(parent_map).fillna(df['brands'])
    return df

def fetch_demographics(api_key, region):
    if not api_key:
        return None
    c = Census(api_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')

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

    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    p_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    p_den = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (p_num / p_den.replace(0, 1)) * 100
    return df[df['income'] > 0]

def process_trends(files):
    if not files:
        return "No trend PDFs. Use general training knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() or "" for page in reader.pages[:3]])
        except:
            pass
    return text[:15000]

# =============================================================================
# 4B) EVIDENCE PACK HELPERS (FOR ROBUST STRATEGY PROMPT) + READABLE BULLETS
# =============================================================================
def _clean_str(x, max_len=240):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len]

def build_entity_evidence(df, entity, n=10):
    """
    Builds a compact evidence pack: top N products + claims + ingredient snippets.
    This makes strategy outputs feel 'real' instead of generic.
    """
    g = df[df["parent_company"] == entity].copy()
    if g.empty:
        return []

    if "unique_scans_n" in g.columns:
        g["unique_scans_n"] = pd.to_numeric(g["unique_scans_n"], errors="coerce").fillna(0)
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
    """Converts evidence list into a short text block for the prompt (token-safe)."""
    if not evidence_items:
        return "No evidence available."
    lines = []
    for it in evidence_items[:10]:
        lines.append(
            f"- {it['product_name']} | Claims: {it['claims_tags']} | Ing: {it['ingredients_snip']}"
        )
    return "\n".join(lines)

def bullets_html(xs):
    """Render list -> HTML bullets for Streamlit to_html() display."""
    if xs is None:
        return ""
    if isinstance(xs, list):
        return "<br>".join([f"• {re.sub(r'<','&lt;', str(x))}" for x in xs if str(x).strip()])
    return re.sub(r"<", "&lt;", str(xs))

# =============================================================================
# 5) SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🤖 Strategy Agent")
    GEMINI_API = st.text_input("Gemini API Key", type="password")
    CENSUS_API = st.text_input("Census API Key", type="password")
    st.divider()
    TARGET_REGION = st.selectbox("Strategic Region", list(REGION_MAP.keys()))
    TARGET_CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    uploaded_files = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    execute = st.button("🚀 Run Market Scan", type="primary")

# =============================================================================
# 6) MAIN DASHBOARD
# =============================================================================
st.title("🚀 Strategic Intelligence Hub")

if execute and GEMINI_API:
    with st.status("⚙️ Agent Working...", expanded=True) as status:
        st.session_state.demographics_df = fetch_demographics(CENSUS_API, TARGET_REGION)
        st.session_state.market_df = fetch_market_intelligence(TARGET_CATEGORY, GEMINI_API)
        st.session_state.trends_text = process_trends(uploaded_files)
        st.session_state.data_fetched = True
        status.update(label="✅ Data Acquisition Complete", state="complete")

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df

    if m_df is None or m_df.empty:
        st.error("❌ Market Data Error: No items returned.")
        st.stop()
    if d_df is None or d_df.empty:
        st.error("❌ Census Error: No data returned.")
        st.stop()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Market SKUs", len(m_df))
    parent_list = sorted(m_df['parent_company'].dropna().unique().tolist())
    kpi2.metric("Clean Parent Entities", len(parent_list))
    kpi3.metric("Avg Income", f"${d_df['income'].mean():,.0f}")

    st.markdown('<div class="section-header">Competitive Landscape Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        my_brand = st.selectbox("Select Your Brand Focus:", parent_list)
        entity_skus = m_df[m_df['parent_company'] == my_brand]
        st.info(f"**{my_brand}** controls **{len(entity_skus)} SKUs**.")

    with col_r:
        st.bar_chart(m_df['parent_company'].value_counts().head(10))

    # =============================================================================
    # STRATEGIC DIRECTIVE ENGINE (ANALYSIS ON GEMINI-2.5-PRO; CLEANING ON 3-FLASH-PREVIEW)
    # =============================================================================
    st.divider()
    if st.button("✨ Generate Full Strategic Directive", type="primary"):
        with st.spinner("🧠 Synthesizing Strategy via Gemini-2.5-Pro (with SKU evidence)..."):
            genai.configure(api_key=GEMINI_API)
            model = genai.GenerativeModel("gemini-2.5-pro")

            # Top competitors by entity presence in the dataset
            comp_list = (
                m_df[m_df["parent_company"] != my_brand]["parent_company"]
                .value_counts()
                .head(6)
                .index
                .tolist()
            )

            # Evidence packs
            my_evidence = build_entity_evidence(m_df, my_brand, n=10)
            comp_evidence_map = {c: build_entity_evidence(m_df, c, n=8) for c in comp_list}

            my_evidence_txt = summarize_entity_signals(my_evidence)
            comp_evidence_txt = "\n\n".join(
                [f"{c}:\n{summarize_entity_signals(comp_evidence_map.get(c, []))}" for c in comp_list]
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
- Category: {TARGET_CATEGORY}
- Region: {TARGET_REGION}
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
                response = model.generate_content(prompt)
                res_txt = re.sub(r'```json\s?|```', '', response.text).strip()
                result = json.loads(res_txt)

                # ----------------------------
                # RENDER: EXECUTIVE SUMMARY
                # ----------------------------
                st.markdown("## 📋 Executive Summary")
                es = result.get("executive_summary", {})
                st.info(es.get("bluf", ""))

                c_es1, c_es2 = st.columns(2)
                with c_es1:
                    st.markdown("**What we know (evidence-based)**")
                    for b in es.get("what_we_know", []):
                        st.write(f"• {b}")
                with c_es2:
                    st.markdown("**What we don’t know (risks / gaps)**")
                    for b in es.get("what_we_dont_know", []):
                        st.write(f"• {b}")

                # ----------------------------
                # RENDER: MARKET STRUCTURE
                # ----------------------------
                st.markdown("## 🧱 Market Structure")
                ms = result.get("market_structure", {})
                bpl = ms.get("branded_vs_private_label", {})

                st.markdown("**Branded vs Private Label**")
                c_ms1, c_ms2, c_ms3 = st.columns(3)
                with c_ms1:
                    st.markdown("_Observations_")
                    for b in bpl.get("observations", []):
                        st.write(f"• {b}")
                with c_ms2:
                    st.markdown("_Implications_")
                    for b in bpl.get("implications", []):
                        st.write(f"• {b}")
                with c_ms3:
                    st.markdown("_Watchouts_")
                    for b in bpl.get("watchouts", []):
                        st.write(f"• {b}")

                arena = ms.get("competitive_arena", [])
                if arena:
                    st.markdown("**Competitive Arena**")
                    st.write(pd.DataFrame(arena).to_html(index=False), unsafe_allow_html=True)

                # ----------------------------
                # RENDER: OCCASION CARDS
                # ----------------------------
                st.markdown("## 🎯 Occasion Cards (MECE)")
                for card in result.get("occasion_cards", []):
                    occ_name = card.get("occasion_name", "Occasion")
                    headline = card.get("slide_headline", "")
                    with st.expander(f"🧩 {occ_name} — {headline}", expanded=True):
                        st.markdown(f"**Definition:** {card.get('definition','')}")
                        st.markdown(f"**Who wins today:** {card.get('who_wins_today','')}")
                        st.markdown("**Winning offer (evidence cues)**")
                        for b in card.get("winning_offer", []):
                            st.write(f"• {b}")

                        st.markdown(f"**Gap for {my_brand}**")
                        for b in card.get(f"gap_for_{my_brand}", []):
                            st.write(f"• {b}")

                        st.markdown(f"**Moves for {my_brand} (actionable)**")
                        for b in card.get(f"moves_for_{my_brand}", []):
                            st.write(f"• {b}")

                # ----------------------------
                # RENDER: CLAIMS STRATEGY
                # ----------------------------
                st.markdown("## 🏷️ Claims Strategy")
                cs = result.get("claims_strategy", {})
                c_cs1, c_cs2 = st.columns(2)
                with c_cs1:
                    st.markdown("**Category claims that win**")
                    for b in cs.get("category_claims_that_win", []):
                        st.write(f"• {b}")
                with c_cs2:
                    st.markdown(f"**Opportunity claims for {my_brand}**")
                    for b in cs.get(f"opportunity_claims_for_{my_brand}", []):
                        st.write(f"• {b}")

                patterns = cs.get("competitor_claim_patterns", [])
                if patterns:
                    st.markdown("**Competitor claim patterns (with proof points)**")
                    st.write(pd.DataFrame(patterns).to_html(index=False), unsafe_allow_html=True)

                # ----------------------------
                # RENDER: INGREDIENT AUDIT (READABLE)
                # ----------------------------
                st.markdown("## 🔬 Ingredient Audit (Readable)")
                ing = result.get("ingredient_audit", [])
                if ing:
                    ing_df = pd.DataFrame(ing)

                    # Render {my_brand} list column as bullets if present
                    if my_brand in ing_df.columns:
                        ing_df[my_brand] = ing_df[my_brand].apply(bullets_html)

                    # competitor_1 / competitor_2 are dicts; convert to readable HTML blocks
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

                # ----------------------------
                # RENDER: STRATEGIC QUESTIONS
                # ----------------------------
                st.markdown("## 🧐 Strategic Questions")
                for q in result.get("strategic_questions", []):
                    st.warning(f"👉 {q}")

            except Exception as e:
                st.error(f"Analysis Failed: {e}")

    # =============================================================================
    # AI Normalization Audit (shows raw->parent mapping)
    # =============================================================================
    with st.expander("🔍 AI Normalization Audit"):
        st.dataframe(m_df[['brands', 'parent_company']].drop_duplicates())
