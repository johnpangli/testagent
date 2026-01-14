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
    div.block-container {padding-top: 1.5rem; max-width: 100%;}
    
    /* Professional Color Palette */
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --light-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #1e293b;
    }
    
    /* The Red Button Styling */
    div.stButton > button:first-child[kind="primary"] {
        background-color: #dc2626 !important;
        border-color: #dc2626 !important;
        color: white !important;
        width: 100%; 
        border-radius: 6px; 
        font-weight: 700;
        padding: 0.8rem 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(220, 38, 38, 0.3);
    }
    
    /* Metrics Dashboard */
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 800;
        color: var(--primary-blue);
    }
    
    /* Professional Tables */
    table {
        width: 100%; 
        border-collapse: collapse;
        font-size: 13px;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
    }
    th {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white !important;
        text-align: left;
        padding: 12px 15px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
    }
    td { 
        padding: 12px 15px; 
        border-bottom: 1px solid var(--border-gray);
    }
    
    /* Insight Cards for AI Output */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin: 15px 0;
        border-left: 6px solid var(--accent-blue);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .insight-card h4 {
        color: var(--primary-blue);
        margin-top: 0;
        font-size: 18px;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
    }
    
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 14px 24px;
        border-radius: 8px;
        margin: 25px 0 15px 0;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
    """Parses JSON even if LLM includes markdown wrappers or trailing commas."""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    try:
        return json.loads(text)
    except:
        try:
            # Handle common LLM formatting errors
            text = re.sub(r',\s*\}', '}', text)
            text = re.sub(r',\s*\]', ']', text)
            return json.loads(text)
        except:
            return {}

def ai_brand_cleaner(df, api_key):
    """Uses Gemini to unify fragmented brand names (e.g., 'KRAFT' vs 'Kraft Foods')."""
    if df.empty or not api_key: return df
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    unique_brands = df['brands'].dropna().unique().tolist()[:60]
    prompt = f"Clean this list of CPG brands. Group variations into one official name. Return JSON ONLY: {{'brand_map': {{'Messy': 'Clean'}} }}. List: {unique_brands}"
    try:
        response = model.generate_content(prompt)
        mapping = safe_json_parse(response.text).get('brand_map', {})
        df['brand_clean'] = df['brands'].apply(lambda x: mapping.get(x, x)).astype(str).str.title().str.strip()
        return df
    except:
        df['brand_clean'] = df['brands']
        return df

# --- 4. DATA PIPELINES ---

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
    """Fetches high-depth Census data including Age, Households, and Poverty."""
    if not api_key: return None
    c = Census(api_key)
    state_codes = REGION_MAP.get(region_input, [region_input.upper()])
    target_states = [us.states.lookup(s) for s in state_codes if us.states.lookup(s)]
    
    if not target_states: return None
    
    # B01003_001E: Pop | B19013_001E: Income | B17001_002E: Poverty | B01002_001E: Age
    # B11001_001E: Total HH | B11001_002E: Family HH
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
    
    # Process Metrics
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
    
    return df[(df['income'] > 0) & (df['population'] > 1000)].sort_values(['population'], ascending=False).head(20)

def get_market_data(cat_input, gemini_key):
    """Pulls product-level data from OpenFoodFacts."""
    tech_tag = CATEGORY_MAP.get(cat_input, cat_input.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    all_prods = []
    for page in range(1, 4):
        p = {"action": "process", "tagtype_0": "categories", "tag_contains_0": "contains", "tag_0": tech_tag, 
             "json": "1", "page_size": 100, "page": page, "cc": "us",
             "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n"}
        try:
            r = requests.get(url, params=p, timeout=15)
            d = r.json().get('products', [])
            if not d: break
            all_prods.extend(d)
        except: break
    df = pd.DataFrame(all_prods)
    if not df.empty:
        df = df.drop_duplicates(subset=['product_name'])
        df = ai_brand_cleaner(df, gemini_key)
    return df

def process_trends(files):
    """Extracts text from uploaded PDF intelligence reports."""
    if not files: return "General market knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([p.extract_text() for p in reader.pages[:3]])
        except: pass
    return text[:15000]

# --- 5. SIDEBAR & DATA ORCHESTRATION ---
with st.sidebar:
    st.header("Hub Configuration")
    GEM_KEY = st.text_input("Gemini API Key", type="password")
    CEN_KEY = st.text_input("Census API Key", type="password")
    st.markdown("---")
    SEL_REGION = st.selectbox("Market Region", list(REGION_MAP.keys()))
    SEL_CAT = st.selectbox("Category", list(CATEGORY_MAP.keys()))
    PDFS = st.file_uploader("Intelligence Files", type=['pdf'], accept_multiple_files=True)
    
    if st.button("▶ Start Intelligence Collection", use_container_width=True):
        if not GEM_KEY or not CEN_KEY:
            st.warning("Please provide API Keys.")
        else:
            with st.status("Gathering Intelligence...", expanded=True) as status:
                st.write("Fetching Demographics (Age, Income, HH Type)...")
                st.session_state.demographics_df = get_demographics(CEN_KEY, SEL_REGION)
                st.write("Scraping Market Landscape...")
                st.session_state.market_df = get_market_data(SEL_CAT, GEM_KEY)
                st.write("Parsing PDFs...")
                st.session_state.trends_text = process_trends(PDFS)
                st.session_state.data_fetched = True
                st.session_state.analysis_generated = False
                status.update(label="Collection Complete", state="complete")

# --- 6. AGENTIC DASHBOARD ---
st.title("Strategic Intelligence Hub")
st.caption("AI-Powered CPG Market & Demographic Disruption Analysis")

if not st.session_state.data_fetched:
    st.info("Input credentials and parameters in the sidebar to begin.")
else:
    d_df = st.session_state.demographics_df
    m_df = st.session_state.market_df

    # --- THE DYNAMIC LAYOUT SHIFT ---
    if st.session_state.analysis_generated:
        left_pane, right_report = st.columns([1, 2.8], gap="large")
    else:
        left_pane = st.container()
        right_report = None

    with left_pane:
        # Market Overview Module
        st.markdown('<div class="section-header">Demographic Context</div>', unsafe_allow_html=True)
        met1, met2 = st.columns(2)
        met1.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
        met2.metric("Median Age", f"{d_df['median_age'].mean():.1f}")
        
        met3, met4 = st.columns(2)
        met3.metric("% Family HH", f"{d_df['family_pct'].mean():.1f}%")
        met4.metric("% Single HH", f"{d_df['single_pct'].mean():.1f}%")
        
        st.metric("Total Population Base", f"{d_df['population'].sum():,.0f}")

        # Strategy Selection Module
        st.markdown('<div class="section-header">Strategic Parameters</div>', unsafe_allow_html=True)
        brand_opts = sorted(m_df['brand_clean'].unique().tolist())
        target_brand = st.selectbox("Select Target Brand", brand_opts)
        
        st.markdown("**Select Analysis Modules**")
        run_occ = st.checkbox("Occasion Matrix", value=True)
        run_clm = st.checkbox("Positioning Strategy", value=True)
        run_qst = st.checkbox("Leadership Interrogation", value=True)
        run_ing = st.checkbox("Technical Ingredient Audit", value=False)
        
        # THE RED BUTTON
        if st.button("▶ Generate Strategic Analysis", type="primary"):
            with st.spinner("Agent computing disruption report..."):
                genai.configure(api_key=GEM_KEY)
                agent = genai.GenerativeModel('gemini-1.5-pro')
                
                comp_set = m_df[m_df['brand_clean']!=target_brand].groupby('brand_clean')['unique_scans_n'].sum().sort_values(ascending=False).head(3).index.tolist()
                
                prompt = f"""
                ROLE: Chief Strategy Officer. 
                REGION: {SEL_REGION} | CATEGORY: {SEL_CAT} | BRAND: {target_brand}
                
                DEMOGRAPHICS:
                - Avg Age: {d_df['median_age'].mean():.1f}
                - Household Mix: {d_df['family_pct'].mean():.1f}% Family / {d_df['single_pct'].mean():.1f}% Single-person.
                - Affordability: {d_df['poverty_rate'].mean():.1f}% Poverty Rate.
                
                TASK: Generate a high-stakes JSON strategy report. 
                Ensure the 'occasions_matrix' maps specific consumer life-stages (based on age/household data) to packaging needs.
                Include 'executive_summary', 'occasions_matrix' (list), 'claims_strategy', 'strategic_questions'.
                """
                try:
                    resp = agent.generate_content(prompt)
                    st.session_state.result_json = safe_json_parse(resp.text)
                    st.session_state.analysis_generated = True
                    st.rerun()
                except Exception as err:
                    st.error(f"Agent Fault: {err}")

    # --- THE GENERATED REPORT PANEL ---
    if st.session_state.analysis_generated and right_report:
        with right_report:
            data = st.session_state.result_json
            st.markdown('<div class="section-header">Executive Strategic Intelligence</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>Strategic Summary</h4>
                <p>{data.get("executive_summary", "Strategic data compiled.")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "occasions_matrix" in data:
                st.subheader("Disruption Occasion Matrix")
                o_df = pd.DataFrame(data["occasions_matrix"])
                st.write(o_df.to_html(index=False, escape=False), unsafe_allow_html=True)
            
            if "claims_strategy" in data:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>Claim & Positioning Delta</h4>
                    <p>{str(data["claims_strategy"])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if "strategic_questions" in data:
                st.subheader("Critical Interrogations for Leadership")
                for q in data["strategic_questions"]:
                    st.warning(f"**Question:** {q}")

# --- 7. FOOTER ---
st.markdown("---")
st.caption("Strategic Intelligence Hub | Built for Enterprise Strategy | Data: US Census ACS & OpenFoodFacts")
