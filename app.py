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
# 1) PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Product Intelligence Hub", page_icon="■", layout="wide")

st.markdown("""
<style>
/* ----------------- Layout & typography ----------------- */
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
div.block-container {
  padding-top: 3.25rem !important;
  max-width: 1400px;
}

/* ----------------- Streamlit chrome (light) ----------------- */
header[data-testid="stHeader"] {
  background: #f6f7fb !important;
  border-bottom: 1px solid #e5e7eb !important;
}
div[data-testid="stToolbar"] { background: #f6f7fb !important; }
div[data-testid="stDecoration"] { background: #f6f7fb !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* App background */
.stApp { background: #f6f7fb; color: #0f172a; }

/* Sidebar background */
section[data-testid="stSidebar"] {
  background: #eef2f7 !important;
  border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * { color: #0f172a !important; }

/* Sidebar widget input backgrounds */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="textarea"] > div {
  background: #dde3ea !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder { color: #64748b !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #0f172a !important; }

/* Headings */
h1 { font-size: 24px !important; margin: 0.1rem 0 0.25rem 0; letter-spacing: -0.02em; }
h2 { font-size: 16px !important; margin: 0.85rem 0 0.25rem 0; letter-spacing: -0.02em; }
h3 { font-size: 14px !important; margin: 0.65rem 0 0.25rem 0; letter-spacing: -0.02em; }

.small-muted { color: #64748b !important; font-size: 12px; line-height: 1.35; }
.hr { height: 1px; background: #e5e7eb; margin: 12px 0; }

/* Bring back spacing between rows (less squish) */
div[data-testid="stVerticalBlock"] { gap: 0.70rem; }

/* Section label */
.section-label {
  font-size: 11.5px;
  font-weight: 800;
  color: #334155;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 10px 0 8px 2px;
}

/* Buttons base (keep clean for everything else) */
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

/* Primary buttons light */
button[kind="primary"] {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
}
button[kind="primary"]:hover {
  background: #f8fafc !important;
  border-color: #94a3b8 !important;
}

/* Cards / tiles */
.card {
  background: #ffffff;
  border: 1px solid #e6e8ef;
  border-radius: 16px;
  padding: 12px 14px 10px 14px;
  box-shadow: 0 1px 0 rgba(15,23,42,0.04);
}

/* Row wrapper so we can add breathing room between tiles */
.tile-row {
  margin-bottom: 10px;
}

/* Card header row: title left, button right */
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 2px;
}

.card-title {
  font-size: 16px;
  font-weight: 850;
  color: #0f172a;
  margin: 0;
  letter-spacing: -0.02em;
}

.card-sub {
  font-size: 12px;
  color: #64748b;
  margin: 0 0 6px 0;
}

.card-bullets {
  font-size: 13px;
  color: #0f172a;
  line-height: 1.28; /* slightly more breathable */
}

.card-bullets div { margin: 4px 0; }

/* --- THE CLEAN DARK GREY PILL (scoped) --- */
.tile-action div[data-testid="stButton"]{
  width: 100% !important;
  display: flex !important;
  justify-content: flex-end !important;  /* push pill to the right */
}

.tile-action .stButton>button{
  width: auto !important;                /* override global 100% */
  min-width: 0 !important;
  border-radius: 9999px !important;      /* true pill */
  padding: 6px 12px !important;
  border: 1px solid #9aa4b2 !important; /* darker border */
  background: #b1bdcf !important;        /* darker grey fill */
  color: #0f172a !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
  line-height: 1 !important;
}

.tile-action .stButton>button:hover{
  background: #b8c4d6 !important;        /* darker hover */
  border-color: #7f8a99 !important;
}

/* Tables */
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

/* Expander light */
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
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("data_fetched", False)
ss_init("market_df", None)
ss_init("demographics_df", None)
ss_init("trends_text", "")

# IMPORTANT: persist keys so locked state doesn't lose them on rerun
ss_init("gemini_api_key", "")
ss_init("census_api_key", "")

ss_init("ui_locked", False)
ss_init("sel_region", "Midwest")
ss_init("sel_category", "Snack Nuts")
ss_init("sel_focus", None)

ss_init("sel_branded_peers", [])
ss_init("sel_private_label_peers", [])
ss_init("sel_all_peers", [])

ss_init("directive_result", None)

# Debug / troubleshooting
ss_init("last_error", "")
ss_init("last_gemini_raw", "")
ss_init("last_gemini_clean", "")
ss_init("last_prompt_chars", 0)
ss_init("last_model", "")

# =============================================================================
# 3) GEMINI RESPONSE TEXT SAFE EXTRACTOR
# =============================================================================
def _safe_response_text(resp) -> str:
    # Prefer .text if present
    try:
        t = (resp.text or "").strip()
        if t:
            return t
    except Exception:
        pass

    # Fallback: candidates/parts
    try:
        texts = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for p in parts:
                    pt = getattr(p, "text", None)
                    if pt:
                        texts.append(pt)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _strip_fences(txt: str) -> str:
    if not txt:
        return ""
    # Remove ```json ... ``` or ``` ... ```
    return re.sub(r"```json\s*|```", "", txt).strip()

def _extract_json_object(txt: str) -> str:
    """
    If the model returned extra text, attempt to isolate the first {...} block.
    """
    if not txt:
        return ""
    s = txt.strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first:last+1]
    return s

def _json_loads_with_debug(txt: str, label: str = "Gemini"):
    """
    Loads JSON with better error messages and stores debug artifacts.
    """
    if not txt or not txt.strip():
        raise ValueError(f"{label} returned an empty string (nothing to parse as JSON).")

    clean = _strip_fences(txt)
    clean = _extract_json_object(clean)

    st.session_state.last_gemini_raw = txt[:4000]
    st.session_state.last_gemini_clean = clean[:4000]

    try:
        return json.loads(clean)
    except json.JSONDecodeError as je:
        # include a small snippet around the failing location
        pos = getattr(je, "pos", None)
        context = ""
        if isinstance(pos, int):
            start = max(0, pos - 160)
            end = min(len(clean), pos + 160)
            context = clean[start:end]
        raise json.JSONDecodeError(
            f"{label} did not return valid JSON. {je.msg}. Context near error:\n{context}",
            je.doc,
            je.pos
        )

# =============================================================================
# 3A) MDM PARENT MAPPING (LOCKED TO GEMINI-3-FLASH-PREVIEW)
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
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = model.generate_content(prompt)
            txt = _safe_response_text(resp)
            if not txt:
                time.sleep(0.8 * attempt)
                continue

            data = _json_loads_with_debug(txt, label="MDM mapping (gemini-3-flash-preview)")
            out = {item["raw"]: item["canonical_parent"] for item in data.get("Mapping", []) if item.get("raw")}
            if out:
                return out
            time.sleep(0.8 * attempt)
        except Exception as e:
            last_exc = e
            time.sleep(0.8 * attempt)
            continue

    st.error("MDM Engine Error: empty/invalid response from Gemini. Falling back to raw brands.")
    if last_exc:
        with st.expander("MDM debug details"):
            st.exception(last_exc)
            if st.session_state.last_gemini_raw:
                st.code(st.session_state.last_gemini_raw)
    return {b: b for b in messy_brands}

# =============================================================================
# 3B) AI-LED ENTITY TYPING: branded vs private_label vs unknown (LOCKED TO 3-FLASH)
# =============================================================================
def classify_entity_types(parent_companies, sample_products_by_parent, api_key):
    if not parent_companies or not api_key:
        return {}

    parent_companies = parent_companies[:90]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")

    evidence_lines = []
    for p in parent_companies:
        samples = (sample_products_by_parent.get(p, []) or [])[:3]
        sample_txt = " | ".join([str(s.get("product_name", "")).strip() for s in samples if s.get("product_name")])[:240]
        evidence_lines.append(f"- {p} :: samples: {sample_txt}")

    prompt = f"""
ACT AS: CPG + Retailer Brand Data Specialist.
TASK: Classify each entity as one of:
- "private_label": retailer-owned / store brand / house brand (incl. retailer corporate parent)
- "branded": manufacturer/CPG owner brand
- "unknown": not enough info; do not guess

RULES:
- Use samples as hints (Kirkland/Great Value/365/etc -> private_label).
- If the entity is clearly a retailer corporate parent or retailer brand -> private_label.
- If ambiguous -> unknown.
- Return JSON only.

ENTITIES:
{chr(10).join(evidence_lines)}

RETURN JSON ONLY:
{{
  "entity_types": [
    {{"parent_company":"...", "entity_type":"branded|private_label|unknown", "rationale":"short"}}
  ]
}}
"""
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = model.generate_content(prompt)
            txt = _safe_response_text(resp)
            if not txt:
                time.sleep(0.8 * attempt)
                continue

            data = _json_loads_with_debug(txt, label="Entity typing (gemini-3-flash-preview)")

            out = {}
            for row in data.get("entity_types", []):
                pc = row.get("parent_company")
                et = row.get("entity_type", "unknown")
                if pc:
                    out[pc] = et

            if out:
                return out

            time.sleep(0.8 * attempt)
        except Exception as e:
            last_exc = e
            time.sleep(0.8 * attempt)
            continue

    if last_exc:
        with st.expander("Entity typing debug details"):
            st.exception(last_exc)
            if st.session_state.last_gemini_raw:
                st.code(st.session_state.last_gemini_raw)

    return {p: "unknown" for p in parent_companies}

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

    last_http = None
    RETRY_SLEEP_SECONDS = 15

    for page in range(1, 6):
        # STREAMLINED: Use requested status text
        status_text.text("Searching/collecting data")

        url = (
            "https://world.openfoodfacts.org/cgi/search.pl?"
            "action=process"
            f"&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}"
            "&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States"
            "&json=1&page_size=100"
            f"&page={page}"
            "&fields=product_name,brands,countries_tags,ingredients_text,labels_tags,unique_scans_n"
        )

        page_products = None

        for attempt in (1, 2):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                last_http = r.status_code

                if r.status_code != 200:
                    if attempt == 1:
                        status_text.text("All Results Found")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    else:
                        status_text.text("Double Checking...")
                        page_products = None
                        break

                try:
                    payload = r.json()
                except Exception:
                    if attempt == 1:
                        status_text.text("All Results Found")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    else:
                        status_text.text("Double Checking...")
                        page_products = None
                        break

                products = payload.get("products", [])

                if not products:
                    if attempt == 1:
                        status_text.text("All Results Found")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    else:
                        status_text.text("Double Checking...")
                        page_products = None
                        break

                # Success
                page_products = products
                break

            except Exception as e:
                if attempt == 1:
                    status_text.text("All Results Found")
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue
                else:
                    status_text.text("Double Checking...")
                    page_products = None
                    break

        if not page_products:
            # Final state display
            status_text.text("Done ✅")
            break

        all_products.extend(page_products)
        time.sleep(0.45) 

    # Clean up status text after loop
    status_text.empty()

    df = pd.DataFrame(all_products)
    if df.empty:
        st.warning(f"No products returned from OpenFoodFacts for tag '{tech_tag}'. Last HTTP: {last_http}")
        return df

    df["brands"] = df["brands"].astype(str).str.strip().str.strip(",")
    df = df[~df["brands"].isin(["nan", "None", "", "Unknown", "null"])]
    df = df.drop_duplicates(subset=["product_name"])
    df["unique_scans_n"] = pd.to_numeric(df.get("unique_scans_n"), errors="coerce").fillna(0)

    unique_messy = df["brands"].unique().tolist()
    with st.spinner(f"Normalizing corporate parents ({len(unique_messy)} brands)…"):
        parent_map = get_canonical_parent_map(unique_messy, gemini_key)

    df["parent_company"] = df["brands"].map(parent_map).fillna(df["brands"])

    parents = df["parent_company"].dropna().unique().tolist()
    sample_by_parent = {}
    for p in parents:
        tmp = df[df["parent_company"] == p].sort_values("unique_scans_n", ascending=False).head(3)
        sample_by_parent[p] = tmp[["product_name"]].to_dict("records")

    with st.spinner("Tagging entities (branded vs private label)…"):
        type_map = classify_entity_types(parents, sample_by_parent, gemini_key)

    df["entity_type"] = df["parent_company"].map(type_map).fillna("unknown")
    return df

def fetch_demographics(census_key, region):
    if not census_key:
        return None

    c = Census(census_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []

    vars = (
        "B01003_001E",
        "B19013_001E",
        "B17001_002E",
        "B17001_001E",
        "B01002_001E",
        "B11001_001E",
        "B11001_002E",
        "B11001_007E",
        "B25010_001E",
        "B09001_001E",
    )

    last_exc = None
    for s_code in states:
        try:
            state_obj = us.states.lookup(s_code)
            if not state_obj:
                continue
            res = c.acs5.state_zipcode(vars, state_obj.fips, Census.ALL)
            all_data.extend(res)
        except Exception as e:
            last_exc = e
            continue

    df = pd.DataFrame(all_data)
    if df.empty:
        st.warning("Census returned no rows for the selected region/states.")
        if last_exc:
            with st.expander("Census debug details"):
                st.exception(last_exc)
        return None

    df["population"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    df["income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")

    p_num = pd.to_numeric(df["B17001_002E"], errors="coerce")
    p_den = pd.to_numeric(df["B17001_001E"], errors="coerce")
    df["poverty_rate"] = (p_num / p_den.replace(0, 1)) * 100

    df["median_age"] = pd.to_numeric(df["B01002_001E"], errors="coerce")

    hh_total = pd.to_numeric(df["B11001_001E"], errors="coerce").replace(0, 1)
    df["family_household_pct"] = pd.to_numeric(df["B11001_002E"], errors="coerce") / hh_total * 100
    df["nonfamily_household_pct"] = pd.to_numeric(df["B11001_007E"], errors="coerce") / hh_total * 100

    df["avg_household_size"] = pd.to_numeric(df["B25010_001E"], errors="coerce")

    under18 = pd.to_numeric(df["B09001_001E"], errors="coerce")
    df["under18_share_pct"] = (under18 / df["population"].replace(0, 1)) * 100

    df = df[(df["income"] > 0) & (df["population"] > 0)]
    return df

def process_trends(files):
    if not files:
        return ""
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([(page.extract_text() or "") for page in reader.pages[:3]])
        except Exception:
            pass
    return text[:15000]

# =============================================================================
# 5) HELPERS
# =============================================================================
def html_table(df: pd.DataFrame, max_rows: int = 250):
    if df is None or df.empty:
        st.markdown("<div class='small-muted'>No rows.</div>", unsafe_allow_html=True)
        return
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
        st.markdown(f"<div class='small-muted'>Showing first {max_rows} rows.</div>", unsafe_allow_html=True)
    st.write(df2.to_html(index=False, escape=True), unsafe_allow_html=True)

def _clean_str(x, max_len=220):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len]

def build_entity_evidence(df, entity, n=10):
    g = df[df["parent_company"] == entity].copy()
    if g.empty:
        return []
    g = g.sort_values("unique_scans_n", ascending=False).dropna(subset=["product_name"]).head(n)
    items = []
    for _, r in g.iterrows():
        items.append({
            "product_name": _clean_str(r.get("product_name"), 120),
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

def choose_peers(m_df, focus_parent, branded_n=4, pl_n=2):
    peers_df = m_df[m_df["parent_company"] != focus_parent].copy()
    peers_df["w"] = peers_df["unique_scans_n"].fillna(0)

    branded = (
        peers_df[peers_df["entity_type"] == "branded"]
        .groupby("parent_company")["w"].sum()
        .sort_values(ascending=False)
        .head(branded_n).index.tolist()
    )
    private_label = (
        peers_df[peers_df["entity_type"] == "private_label"]
        .groupby("parent_company")["w"].sum()
        .sort_values(ascending=False)
        .head(pl_n).index.tolist()
    )

    need = (branded_n - len(branded)) + (pl_n - len(private_label))
    backfill = []
    if need > 0:
        backfill = (
            peers_df[~peers_df["parent_company"].isin(set(branded + private_label))]
            .groupby("parent_company")["w"].sum()
            .sort_values(ascending=False)
            .head(need).index.tolist()
        )

    return branded, private_label, (branded + private_label + backfill)

def _truncate(s, n=140):
    s = str(s).strip()
    return s if len(s) <= n else (s[: n - 1].rstrip() + "…")

# =============================================================================
# 6) DETAILS RENDERER
# =============================================================================
def render_details(panel_key, result, my_brand, m_df, d_df):
    if not result:
        st.write("No directive generated yet.")
        return

    es = result.get("executive_summary", {})
    ms = result.get("market_structure", {})
    bpl = ms.get("branded_vs_private_label", [])
    roles = ms.get("competitive_roles", [])
    occ = result.get("occasion_cards", [])
    cs = result.get("claims_strategy", {})
    ing = result.get("ingredient_audit", [])
    questions = result.get("strategic_questions", [])

    st.markdown("### Details")
    st.markdown("<div class='small-muted'>Observation → Evidence → Inference → Implication</div>", unsafe_allow_html=True)

    if panel_key == "exec":
        st.info(es.get("bluf", ""))
        for i, it in enumerate(es.get("key_insights", []), start=1):
            with st.expander(f"Insight {i}", expanded=(i == 1)):
                st.markdown(f"**Observation:** {it.get('observation','')}")
                st.markdown("**Evidence:**")
                for ev in it.get("evidence", []):
                    st.write(f"• {ev}")
                st.markdown(f"**Inference:** {it.get('inference','')}")
                st.markdown(f"**Implication:** {it.get('implication','')}")
        if es.get("gaps_and_risks"):
            st.markdown("**Gaps & risks**")
            for g in es.get("gaps_and_risks", []):
                st.write(f"• {g}")

    elif panel_key == "market":
        st.markdown("#### Branded vs private label dynamics")
        for i, it in enumerate(bpl, start=1):
            with st.expander(f"Dynamic {i}", expanded=(i == 1)):
                st.markdown(f"**Observation:** {it.get('observation','')}")
                st.markdown("**Evidence:**")
                for ev in it.get("evidence", []):
                    st.write(f"• {ev}")
                st.markdown(f"**Inference:** {it.get('inference','')}")
                st.markdown(f"**Implication:** {it.get('implication','')}")
        if roles:
            st.markdown("#### Competitive roles")
            html_table(pd.DataFrame(roles))

    elif panel_key == "occasions":
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

    elif panel_key == "claims":
        st.markdown("#### Category claim patterns")
        for i, p in enumerate(cs.get("category_claim_patterns", []), start=1):
            with st.expander(f"Pattern {i}: {p.get('pattern','')}", expanded=(i == 1)):
                st.markdown("**Evidence:**")
                for ev in p.get("evidence", []):
                    st.write(f"• {ev}")
                st.markdown(f"**Inference:** {p.get('inference','')}")
                st.markdown(f"**Implication:** {p.get('implication','')}")

        st.markdown("#### Opportunity claims (feasibility-checked)")
        opp = cs.get(f"opportunity_claims_for_{my_brand}", [])
        if opp:
            html_table(pd.DataFrame(opp))

    elif panel_key == "ingredients":
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

    elif panel_key == "mdm":
        st.markdown("#### Entity normalization audit")
        st.markdown("<div class='small-muted'>Spot-check surprising parents before presenting entity counts.</div>", unsafe_allow_html=True)
        audit_df = m_df[["brands", "parent_company", "entity_type"]].drop_duplicates().sort_values(["entity_type", "parent_company", "brands"])
        html_table(audit_df, max_rows=300)

    elif panel_key == "questions":
        st.markdown("#### Strategic questions")
        for q in questions:
            st.write(f"• {q}")

    else:
        st.write("Unknown panel.")

# =============================================================================
# 7) MODAL OPEN
# =============================================================================
def open_details(panel_key, result, my_brand, m_df, d_df):
    if hasattr(st, "dialog"):
        @st.dialog("Details", width="large")
        def _dlg():
            render_details(panel_key, result, my_brand, m_df, d_df)
        _dlg()
    else:
        with st.sidebar:
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown("## Details")
            render_details(panel_key, result, my_brand, m_df, d_df)

# =============================================================================
# 8) SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## Product Intelligence Agent")
    st.markdown("<div class='small-muted'>Run scan → pick focus → generate → one-page readout.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    execute = False
    uploaded_files = None

    if not st.session_state.ui_locked:
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key
        )
        st.session_state.census_api_key = st.text_input(
            "Census API Key",
            type="password",
            value=st.session_state.census_api_key
        )

        st.session_state.sel_category = st.selectbox(
            "Category",
            list(CATEGORY_MAP.keys()),
            index=list(CATEGORY_MAP.keys()).index(st.session_state.sel_category)
            if st.session_state.sel_category in CATEGORY_MAP else 0
        )
        st.session_state.sel_region = st.selectbox(
            "Region",
            list(REGION_MAP.keys()),
            index=list(REGION_MAP.keys()).index(st.session_state.sel_region)
            if st.session_state.sel_region in REGION_MAP else 0
        )

        uploaded_files = st.file_uploader("Trend PDFs (optional)", type=["pdf"], accept_multiple_files=True)
        execute = st.button("Run scan", type="primary")

        st.markdown("<div class='small-muted'>Cleaning: gemini-3-flash-preview • Analysis: gemini-2.5-pro</div>", unsafe_allow_html=True)

    else:
        st.markdown("### Selections")
        st.write(f"Category: **{st.session_state.sel_category}**")
        st.write(f"Region: **{st.session_state.sel_region}**")
        if st.session_state.sel_focus:
            st.write(f"Focus: **{st.session_state.sel_focus}**")

        if st.session_state.sel_branded_peers:
            st.write("**Branded peers:**")
            st.write(", ".join(st.session_state.sel_branded_peers))
        if st.session_state.sel_private_label_peers:
            st.write("**Private label peers:**")
            st.write(", ".join(st.session_state.sel_private_label_peers))

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        if st.button("Edit selections"):
            st.session_state.ui_locked = False
            st.session_state.directive_result = None
            st.session_state.last_error = ""

    with st.expander("Troubleshoot (debug)", expanded=False):
        st.write({
            "data_fetched": st.session_state.data_fetched,
            "ui_locked": st.session_state.ui_locked,
            "has_gemini_key": bool(st.session_state.gemini_api_key),
            "has_census_key": bool(st.session_state.census_api_key),
            "last_model": st.session_state.last_model,
            "last_prompt_chars": st.session_state.last_prompt_chars,
        })
        if st.session_state.last_error:
            st.markdown("**Last error**")
            st.code(st.session_state.last_error)
        if st.session_state.last_gemini_clean:
            st.markdown("**Last Gemini response (cleaned, first 1200 chars)**")
            st.code(st.session_state.last_gemini_clean[:1200])

# =============================================================================
# 9) MAIN
# =============================================================================
st.markdown("# Product Intelligence Hub")
st.markdown("<div class='small-muted'>One-page tiles. Click View details to open a popup.</div>", unsafe_allow_html=True)

# Run scan
if execute:
    if not st.session_state.gemini_api_key or not st.session_state.census_api_key:
        st.error("Please provide both Gemini and Census API keys.")
        st.stop()

    with st.status("Working…", expanded=True) as status:
        st.write("Fetching demographics…")
        try:
            st.session_state.demographics_df = fetch_demographics(st.session_state.census_api_key, st.session_state.sel_region)
        except Exception as e:
            st.session_state.last_error = f"Census fetch failed: {e}"
            st.error("Census fetch failed.")
            st.exception(e)
            st.stop()

        st.write("Fetching market data + normalizing entities…")
        try:
            st.session_state.market_df = fetch_market_intelligence(st.session_state.sel_category, st.session_state.gemini_api_key)
        except Exception as e:
            st.session_state.last_error = f"Market scan failed: {e}"
            st.error("Market scan failed.")
            st.exception(e)
            st.stop()

        st.write("Ingesting trend PDFs…")
        try:
            st.session_state.trends_text = process_trends(uploaded_files)
        except Exception as e:
            st.session_state.last_error = f"PDF ingest failed: {e}"
            st.warning("Trend PDF ingest failed (continuing without trends).")
            st.exception(e)
            st.session_state.trends_text = ""

        st.session_state.data_fetched = True
        status.update(label="Data ready", state="complete")

# Render if ready
if not st.session_state.data_fetched:
    st.markdown("<div class='small-muted'>Run a scan to begin.</div>", unsafe_allow_html=True)
    st.stop()

m_df = st.session_state.market_df
d_df = st.session_state.demographics_df

if m_df is None or m_df.empty:
    st.error("No market data returned.")
    st.stop()
if d_df is None or d_df.empty:
    st.error("No census data returned.")
    st.stop()

# KPIs
parent_list = sorted(m_df["parent_company"].dropna().unique().tolist())
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("SKUs", len(m_df))
k2.metric("Entities", len(parent_list))
k3.metric("Avg income", f"${d_df['income'].mean():,.0f}")
k4.metric("Poverty", f"{d_df['poverty_rate'].mean():.1f}%")
k5.metric("Median age", f"{d_df['median_age'].mean():.1f}")
k6.metric("Family HH", f"{d_df['family_household_pct'].mean():.1f}%")

st.markdown("<div class='section-label'>Market snapshot</div>", unsafe_allow_html=True)

# Focus selection
if not st.session_state.ui_locked:
    c1, c2 = st.columns([2, 1])
    with c1:
        my_brand = st.selectbox(
            "Focus entity",
            parent_list,
            index=0 if (st.session_state.sel_focus is None or st.session_state.sel_focus not in parent_list)
            else parent_list.index(st.session_state.sel_focus),
        )
        st.session_state.sel_focus = my_brand
    with c2:
        st.markdown("<div class='small-muted' style='padding-top: 1.65rem;'>Pick the focal entity, then generate the readout.</div>", unsafe_allow_html=True)
else:
    my_brand = st.session_state.sel_focus or (parent_list[0] if parent_list else None)

st.markdown("<div class='section-label'>Directive</div>", unsafe_allow_html=True)
c_gen1, c_gen2 = st.columns([1, 3])
with c_gen1:
    generate = st.button("Generate", type="primary", disabled=st.session_state.ui_locked)
with c_gen2:
    st.markdown("<div class='small-muted'>Second-order insights (evidence → inference → implication). Includes feasibility-checked claims.</div>", unsafe_allow_html=True)

# Generate directive
if generate:
    st.session_state.last_error = ""
    st.session_state.last_gemini_raw = ""
    st.session_state.last_gemini_clean = ""

    if not st.session_state.gemini_api_key:
        st.error("Missing Gemini API key. Open 'Edit selections' and re-enter it.")
        st.stop()

    branded_peers, private_label_peers, all_peers = choose_peers(m_df, my_brand, branded_n=4, pl_n=2)
    st.session_state.sel_branded_peers = branded_peers
    st.session_state.sel_private_label_peers = private_label_peers
    st.session_state.sel_all_peers = all_peers

    st.session_state.ui_locked = True

    genai.configure(api_key=st.session_state.gemini_api_key)
    model_name = "gemini-2.5-pro"
    st.session_state.last_model = model_name
    model = genai.GenerativeModel(model_name)

    my_evidence_txt = summarize_entity_signals(build_entity_evidence(m_df, my_brand, n=10))
    comp_evidence_txt = "\n\n".join(
        [f"{c}:\n{summarize_entity_signals(build_entity_evidence(m_df, c, n=8))}" for c in all_peers]
    )

    demo_summary = {
        "avg_income": float(d_df["income"].mean()),
        "poverty": float(d_df["poverty_rate"].mean()),
        "median_age": float(d_df["median_age"].mean()),
        "family_hh_pct": float(d_df["family_household_pct"].mean()),
        "nonfamily_hh_pct": float(d_df["nonfamily_household_pct"].mean()),
        "avg_hh_size": float(d_df["avg_household_size"].mean()),
        "under18_share_pct": float(d_df["under18_share_pct"].mean()),
    }

    trends = st.session_state.trends_text or ""

    prompt = f"""
ACT AS: Product Intelligence Lead for a CPG portfolio.
GOAL: Produce "second-order" insights: evidence-led inferences and implications (not obvious observations).

HARD RULES:
1) Each insight must be structured as:
   - "observation": what the dataset shows (1 sentence)
   - "evidence": 2–3 proof points from SKU snippets (explicit examples like product names / claims / ingredient snippets)
   - "inference": what it likely means (1 sentence; causal language)
   - "implication": so what / what decision it changes (1 sentence)
2) Avoid trivialities ("we all sell peanuts"). If obvious, elevate to an implication about positioning, roles, or why it matters.
3) Claims recommendations MUST be feasible:
   - If ingredients/labels indicate a claim is blocked, mark "blocked" with why.
   - Provide "what would need to change" to unlock it (reformulation, certification, substantiation, etc.).
4) If evidence is weak, say so and specify missing data (POS, retailer item file, claims validation, etc.).

CONTEXT:
- Category: {st.session_state.sel_category}
- Region: {st.session_state.sel_region}
- Dataset size: {len(m_df)} SKUs
- Focus entity: {my_brand} ({len(m_df[m_df["parent_company"] == my_brand])} observed SKUs)

DEMOGRAPHICS (ACS 5-year, zip-weighted averages):
- Median income: ${demo_summary["avg_income"]:,.0f}
- Poverty rate: {demo_summary["poverty"]:.1f}%
- Median age: {demo_summary["median_age"]:.1f}
- Family households: {demo_summary["family_hh_pct"]:.1f}%
- Nonfamily households: {demo_summary["nonfamily_hh_pct"]:.1f}%
- Avg household size: {demo_summary["avg_hh_size"]:.2f}
- Under-18 share: {demo_summary["under18_share_pct"]:.1f}%

PEERS (must consider both types):
- Branded peers: {st.session_state.sel_branded_peers}
- Private label peers: {st.session_state.sel_private_label_peers}

EVIDENCE — FOCUS ENTITY SKU SNIPPETS:
{my_evidence_txt}

EVIDENCE — PEERS SKU SNIPPETS:
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
    st.session_state.last_prompt_chars = len(prompt)

    try:
        with st.spinner("Generating…"):
            response = model.generate_content(prompt)

        txt = _safe_response_text(response)
        if not txt:
            st.session_state.last_error = "Gemini returned an empty response."
            st.error("Generation failed: Gemini returned an empty response.")
            st.stop()

        st.session_state.directive_result = _json_loads_with_debug(txt, label=f"Directive generation ({model_name})")

    except Exception as e:
        st.session_state.last_error = str(e)
        st.error("Generation failed.")
        st.exception(e)
        st.stop()

# =============================================================================
# 10) ONE-PAGE VERTICAL TILES
# =============================================================================
result = st.session_state.directive_result
if not result:
    st.markdown("<div class='small-muted'>Generate the readout to see tiles.</div>", unsafe_allow_html=True)
    st.stop()

es = result.get("executive_summary", {})
ms = result.get("market_structure", {})
bpl = ms.get("branded_vs_private_label", [])
occ = result.get("occasion_cards", [])
cs = result.get("claims_strategy", {})
ing = result.get("ingredient_audit", [])

def _tile_from_insights(insights, take=3):
    out = []
    for it in (insights or [])[:take]:
        imp = it.get("implication", "")
        inf = it.get("inference", "")
        obs = it.get("observation", "")
        out.append(imp or inf or obs)
    return [x for x in out if str(x).strip()][:take]

tile_exec = _tile_from_insights(es.get("key_insights", []), 3)
tile_market = _tile_from_insights(bpl, 3)

tile_occ = []
for c in (occ or [])[:3]:
    tile_occ.append(f"{c.get('occasion_name','Occasion')}: {c.get('slide_headline','')}".strip().strip(":"))
tile_occ = [x for x in tile_occ if x][:3]

tile_claims = _tile_from_insights(cs.get("category_claim_patterns", []), 3)

tile_ing = []
for row in (ing or [])[:3]:
    ins = (row.get("insight") or {})
    tile_ing.append(ins.get("implication") or ins.get("inference") or ins.get("observation") or "")
tile_ing = [x for x in tile_ing if x][:3]

tile_mdm = ["Spot-check surprising parents before presenting counts.", "Confirm private label vs branded tags look correct."]

st.markdown("<div class='section-label'>One-page readout</div>", unsafe_allow_html=True)

def tile_row(panel_key, title, subtitle, bullets):
    bullets = [b for b in (bullets or []) if str(b).strip()][:3]
    bullets_html = "".join([f"<div>• {_truncate(b, 150)}</div>" for b in bullets]) or "<div class='small-muted'>No content</div>"

    left, right = st.columns([12, 2], vertical_alignment="top")

    with left:
        st.markdown(f"""
        <div class="tile-row">
          <div class="card">
            <div class="card-header">
              <div class="card-title">{title}</div>
              <div style="width: 0px;"></div>
            </div>
            <div class="card-sub">{subtitle}</div>
            <div class="card-bullets">{bullets_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='tile-action'>", unsafe_allow_html=True)
        if st.button("View details", key=f"{panel_key}_view"):
            open_details(panel_key, result, my_brand, m_df, d_df)
        st.markdown("</div>", unsafe_allow_html=True)

tile_row("exec", "Executive summary", "", tile_exec)
tile_row("market", "Market structure", "Branded vs private label dynamics", tile_market)
tile_row("occasions", "Occasions", "Where value concentrates", tile_occ)
tile_row("claims", "Claims strategy", "Feasible + defensible plays", tile_claims)
tile_row("ingredients", "Ingredient audit", "Drivers of perceived quality / cost", tile_ing)
tile_row("mdm", "Entity normalization", "Validate mappings before presenting", tile_mdm)
