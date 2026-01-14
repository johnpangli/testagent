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

# --- 1. UI CONFIGURATION & PROFESSIONAL CSS ---
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    /* Global Container */
    div.block-container {padding-top: 1.5rem; max-width: 95%;}
    
    /* Professional Color Palette */
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --light-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #1e293b;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 800;
        color: var(--primary-blue);
    }
    
    /* Professional Tables */
    table {
        width: 100%; border-collapse: collapse; font-size: 14px;
        background: white; border-radius: 8px; overflow: hidden;
    }
    th {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white !important; text-align: left; padding: 12px 15px;
        font-weight: 600; text-transform: uppercase; font-size: 11px;
    }
    td { padding: 12px 15px; border-bottom: 1px solid var(--border-gray); }

    /* The Insight Card */
    .insight-card {
        background: white; border-radius: 12px; padding: 20px;
        margin: 15px 0; border-left: 6px solid var(--accent-blue);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .insight-card h4 { color: var(--primary-blue); margin-top: 0; font-size: 17px; border-bottom: 1px solid #eee; padding-bottom: 8px; }

    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 14px 24px; border-radius: 8px;
        margin: 25px 0 15px 0; font-size: 18px; font-weight: 700;
    }

    /* Red Primary Button */
    div.stButton > button:first-child[kind="primary"] {
        background-color: #dc2626 !important; border-color: #dc2626 !important;
        color: white !important; width: 100%; border-radius: 6px; font-weight: 700;
        padding: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
if 'data_fetched' not in st.session_state: st.session_state.data_fetched = False
if 'analysis_generated' not in st.session_state: st.session_state.analysis_generated = False
if 'demographics_df' not in st.session_state: st.session_state.demographics_df = None
if 'market_df' not in st.session_state: st.session_state.market_df = None
if 'trends_text' not in st.session_state: st.session_state.trends_text = ""
if 'result_json' not in st.session_state: st.session_state.result_json = None

# --- 3. REUSABLE UTILITIES ---

def safe_json_parse(text):
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    try:
        return json.loads(text)
    except:
        try:
            text = re.sub(r',\s*\}', '}', text)
            text = re.sub(r',\s*\]', ']', text)
            return json.loads(text)
        except: return {}

def ai_brand_cleaner(df, api_key):
    """Restored Robust Janitor Logic"""
    if df.empty or not api_key: return df
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    unique_brands = df['brands'].dropna().unique().tolist()[:60]
    prompt = f"CPG Data Expert: Merge brand variations (e.g. 'Wrights' & 'Wright Brand' -> 'Wright Brand'). Return JSON ONLY: {{'brand_map': {{'Messy': 'Clean'}} }}. List: {unique_brands}"
    try:
        response = model.generate_content(prompt)
        mapping = safe_json_parse(response.text).get('brand_map', {})
        df['brand_clean'] = df['brands'].apply(lambda x: mapping.get(x, x)).astype(str).str.title().str.strip()
        return df
    except:
        df['brand_clean'] = df['brands'].astype(str).str.title()
        return df

# --- 4. ROBUST DATA PIPELINES (v2.3 Logic) ---

REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA", "CT", "ME", "NH", "VT", "RI"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "SC", "AL", "LA", "MS", "AR", "KY", "WV"],
    "West": ["CA", "WA", "AZ", "CO", "OR", "NV", "UT", "ID", "MT", "WY", "NM"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", "Snack Nuts": "nuts", 
    "Beef Jerky": "meat-snacks", "Coffee": "coffees", "Cereal": "breakfast-cereals", "Chips": "chips"
}

def get_demographics(api_key, region_input):
    """Restored full depth: Age, Income, HH Type, Poverty"""
    if not api_key: return None
    c = Census(api_key)
    state_codes = REGION_MAP.get(region_input, [region_input.upper()])
    target_states = [us.states.lookup(s) for s in state_codes if us.states.lookup(s)]
    
    # B01003: Pop | B19013: Income | B17001: Poverty | B01002: Age | B11001: HH Type
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E', 'B01002_001E', 'B11001_001E', 'B11001_002E')
    
    all_zips = []
    def fetch_task(state):
        try: return c.acs5.state_zipcode(vars, state.fips, Census.ALL)
        except: return []
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_task, s): s for s in target_states}
        for f in as_completed(futures):
            res = f.result()
            if res: all_zips.extend(res)
            
    if not all_zips: return None
    df = pd.DataFrame(all_zips).rename(columns={'zip code tabulation area': 'zip_code'})
    
    # Logic: Direct Metric Calculation
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    df['median_age'] = pd.to_numeric(df['B01002_001E'], errors='coerce')
    
    total_hh = pd.to_numeric(df['B11001_001E'], errors='coerce').replace(0, 1)
    fam_hh = pd.to_numeric(df['B11001_002E'], errors='coerce')
    df['family_pct'] = (fam_hh / total_hh) * 100
    df['single_pct'] = 100 - df['family_pct']
    
    poverty_n = pd.to_numeric(df['B17001_002E'], errors='coerce')
    poverty_d = pd.to_numeric(df['B17001_001E'], errors='coerce').replace(0, 1)
    df['poverty_rate'] = (poverty_n / poverty_d) * 100
    
    return df[(df['income'] > 0) & (df['population'] > 1000)].sort_values(['population'], ascending=False).head(25)

def get_market_data(cat_input, gemini_key):
    """Restored Two-Stage Logic: Tag then Keyword (Fixes 'Bacon' issue)"""
    tech_tag = CATEGORY_MAP.get(cat_input, cat_input.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    all_prods = []
    
    # Stage 1: Tag Search
    for page in range(1, 3):
        p = {"action": "process", "tagtype_0": "categories", "tag_contains_0": "contains", "tag_0": tech_tag, 
             "json": "1", "page_size": 100, "page": page, "cc": "us", "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n"}
        try:
            r = requests.get(url, params=p, timeout=15)
            d = r.json().get('products', [])
            if d: all_prods.extend(d)
        except: break

    # Stage 2: Keyword fallback if result is thin
    if len(all_prods) < 10:
        p_fallback = {"action": "process", "search_terms": cat_input, "json": "1", "page_size": 100, "cc": "us", "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n"}
        try:
            r = requests.get(url, params=p_fallback, timeout=15)
            all_prods.extend(r.json().get('products', []))
        except: pass

    df = pd.DataFrame(all_prods)
    if not df.empty:
        df = df.drop_duplicates(subset=['product_name']).dropna(subset=['brands'])
        df = ai_brand_cleaner(df, gemini_key)
    return df

# --- 5. SIDEBAR & ORCHESTRATION ---
with st.sidebar:
    st.header("Hub Configuration")
    GEM_KEY = st.text_input("Gemini API Key", type="password")
    CEN_KEY = st.text_input("Census API Key", type="password")
    st.markdown("---")
    SEL_REGION = st.selectbox("Market Region", list(REGION_MAP.keys()))
    SEL_CAT = st.selectbox("Category", list(CATEGORY_MAP.keys()))
    PDFS = st.file_uploader("Trend Files (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("▶ Run Intelligence Cycle", use_container_width=True):
        if not GEM_KEY or not CEN_KEY: st.warning("API Keys Required.")
        else:
            with st.status("Gathering Multi-Source Intel...", expanded=True) as status:
                st.write("Census: Pulling Demographic Layers...")
                st.session_state.demographics_df = get_demographics(CEN_KEY, SEL_REGION)
                st.write("Market: Scraping Competitive Landscape...")
                st.session_state.market_df = get_market_data(SEL_CAT, GEM_KEY)
                st.write("Trends: Parsing Intelligence Reports...")
                st.session_state.trends_text = "".join([PdfReader(f).pages[0].extract_text() for f in PDFS]) if PDFS else "General Market Trends."
                st.session_state.data_fetched = True
                st.session_state.analysis_generated = False
                status.update(label="Data Collection Complete", state="complete")

# --- 6. AGENTIC DASHBOARD ---
st.title("Strategic Intelligence Hub")
st.caption("v3.1 | Enterprise CPG Market & Demographic Disruption Analysis")

if st.session_state.data_fetched:
    d_df = st.session_state.demographics_df
    m_df = st.session_state.market_df

    # --- TOP METRIC BAR ---
    st.markdown('<div class="section-header">Regional Market Baseline</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
    m2.metric("Median Age", f"{d_df['median_age'].mean():.1f}")
    m3.metric("% Family", f"{d_df['family_pct'].mean():.1f}%")
    m4.metric("% Poverty", f"{d_df['poverty_rate'].mean():.1f}%")
    m5.metric("Sample Size", f"{len(m_df)} SKUs")

    # --- TWO-PANE LAYOUT ---
    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown('<div class="section-header">Strategic Input</div>', unsafe_allow_html=True)
        # Robust Brand Sorting (Fix for sorting errors)
        brand_opts = sorted([str(b) for b in m_df['brand_clean'].dropna().unique().tolist()])
        target_brand = st.selectbox("Your Target Brand", brand_opts)
        
        st.markdown("**Analysis Modules**")
        run_occ = st.checkbox("Occasion Matrix", value=True)
        run_clm = st.checkbox("Positioning Strategy", value=True)
        run_qst = st.checkbox("Leadership Interrogation", value=True)

        if st.button("▶ Generate Disruption Report", type="primary"):
            with st.spinner("CSO Agent computing strategy..."):
                genai.configure(api_key=GEM_KEY)
                agent = genai.GenerativeModel('gemini-1.5-pro')
                
                # Logic: Inject deep demographic context for the AI
                prompt = f"""
                ROLE: Chief Strategy Officer. REGION: {SEL_REGION} | CATEGORY: {SEL_CAT} | BRAND: {target_brand}
                
                DEMOGRAPHICS:
                - Avg Age: {d_df['median_age'].mean():.1f}
                - HH Mix: {d_df['family_pct'].mean():.1f}% Family / {d_df['single_pct'].mean():.1f}% Single.
                - Affordability: {d_df['poverty_rate'].mean():.1f}% Poverty Rate.
                
                TASK: Generate high-stakes JSON strategy. 
                Crucial: Map the 'occasions_matrix' to consumer life-stages (based on Age/HH Mix) and specific packaging/pricing needs.
                INCLUDE: 'executive_summary', 'occasions_matrix' (list), 'claims_strategy', 'strategic_questions'.
                """
                try:
                    resp = agent.generate_content(prompt)
                    st.session_state.result_json = safe_json_parse(resp.text)
                    st.session_state.analysis_generated = True
                    st.rerun()
                except Exception as e: st.error(f"Agent Error: {e}")

    with right:
        if st.session_state.analysis_generated:
            data = st.session_state.result_json
            st.markdown('<div class="section-header">Executive Strategic Intelligence</div>', unsafe_allow_html=True)
            
            st.markdown(f"""<div class="insight-card"><h4>Strategic Summary</h4><p>{data.get("executive_summary", "")}</p></div>""", unsafe_allow_html=True)
            
            if "occasions_matrix" in data:
                st.subheader("Disruption Occasion Matrix")
                o_df = pd.DataFrame(data["occasions_matrix"])
                st.write(o_df.to_html(index=False, escape=False), unsafe_allow_html=True)

            if "strategic_questions" in data:
                st.subheader("Critical Interrogations")
                for q in data["strategic_questions"]:
                    st.warning(f"**Question:** {q}")
        else:
            st.info("Adjust parameters and click the red button to generate the deep-dive analysis.")



### What's New in v3.1:
1.  **Restored Robust Census Data:** Added back the code to fetch `B01002` (Age) and `B11001` (Household Type). Your metrics bar now shows Family % and Age again.
2.  **Bacon/Peanut Butter Fix:** The `get_market_data` function now uses a "Two-Stage" search. If the technical category tag returns nothing, it falls back to a broad keyword search.
3.  **Type-Safe Brand Sorting:** Added a list comprehension `[str(b) for b in ...]` to the brand selector. This prevents the `TypeError` you saw if the API returns a number or `None` as a brand name.
4.  **CSO Agent Prompting:** The AI prompt is now "Life-Stage Aware." It explicitly tells Gemini to look at the household mix and age to suggest packaging strategies (e.g., "Single-serve for 20% single households").

Would you like me to add a **Data Export** feature so you can download the final analysis as a PDF or Excel file?
