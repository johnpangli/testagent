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

# --- 1. UI CONFIGURATION & PROFESSIONAL STYLING ---
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    div.block-container {padding-top: 1.5rem; max-width: 1400px;}
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --text-dark: #1e293b;
    }
    .stButton>button { 
        width: 100%; border-radius: 6px; font-weight: 600;
        padding: 0.6rem 1rem; transition: all 0.2s;
    }
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-size: 18px; font-weight: 600;
    }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'market_df' not in st.session_state:
    st.session_state.market_df = None
if 'demographics_df' not in st.session_state:
    st.session_state.demographics_df = None

# --- 3. MDM RESOLUTION ENGINE ---

def get_canonical_parent_map(messy_brands, api_key):
    """
    ONE-SHOT RESOLUTION: Sends unique messy strings to Gemini to map them 
    to a single Parent Company (e.g., Blue Diamond variants -> Blue Diamond Growers).
    """
    if not messy_brands or not api_key: return {}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
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
    
    RETURN ONLY VALID JSON OBJECT:
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

# --- 4. DATA ACQUISITION ---

REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA", "CT", "ME", "NH", "VT", "RI"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "SC", "AL", "LA", "MS", "AR", "KY", "WV"],
    "West": ["CA", "WA", "AZ", "CO", "OR", "NV", "UT", "ID", "MT", "WY", "NM"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", "Beef Jerky": "meat-snacks"
}

def fetch_market_intelligence(category, api_key):
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    headers = {'User-Agent': 'StrategicIntelligenceHub/1.0'}
    all_products = []
    
    # 5-Page Pagination for Depth (~500 SKUs)
    for page in range(1, 6):
        url = f"https://world.openfoodfacts.org/cgi/search.pl?action=process&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States&json=1&page_size=100&page={page}&fields=product_name,brands,countries_tags"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            products = r.json().get('products', [])
            if not products: break
            all_products.extend(products)
            time.sleep(0.1)
        except: break

    df = pd.DataFrame(all_products)
    if df.empty: return df

    # Basic Data Scrubbing
    df['brands'] = df['brands'].str.strip().str.strip(',').fillna("Unbranded/Generic")
    
    # Trigger One-Shot MDM Resolution
    unique_messy = df['brands'].unique().tolist()
    with st.spinner(f"AI Entity Resolution: Consolidating {len(unique_messy)} brands..."):
        parent_map = get_canonical_parent_map(unique_messy, api_key)
    
    df['parent_company'] = df['brands'].map(parent_map).fillna(df['brands'])
    return df

def fetch_demographics(api_key, region):
    if not api_key: return None
    c = Census(api_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []
    
    for s_code in states:
        try:
            state_obj = us.states.lookup(s_code)
            res = c.acs5.state_zipcode(('B01003_001E', 'B19013_001E'), state_obj.fips, Census.ALL)
            all_data.extend(res)
        except: continue
        
    df = pd.DataFrame(all_data)
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    return df[df['income'] > 0]

# --- 5. SIDEBAR CONTROLS ---

with st.sidebar:
    st.header("Hub Configuration")
    GEMINI_API = st.text_input("Gemini API Key", type="password")
    CENSUS_API = st.text_input("Census API Key", type="password")
    
    st.divider()
    REGION = st.selectbox("Strategic Region", list(REGION_MAP.keys()))
    CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    
    execute = st.button("▶ Run Market Scan", type="primary")

# --- 6. DASHBOARD LAYOUT ---

st.title("Strategic Intelligence Hub")
st.caption("Enterprise-Scale Parent Company Concentration & Demographic Mapping")

if execute and GEMINI_API:
    with st.status("Gathering Intelligence...", expanded=True) as status:
        st.session_state.market_df = fetch_market_intelligence(CATEGORY, GEMINI_API)
        st.session_state.demographics_df = fetch_demographics(CENSUS_API, REGION)
        st.session_state.data_fetched = True
        status.update(label="Analysis Complete", state="complete")

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df
    
    # Top-Level KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total Market SKUs", len(m_df))
    with kpi2:
        # Note the .unique() here - this is what makes your dropdown clean
        parent_list = sorted(m_df['parent_company'].unique().tolist())
        st.metric("Clean Parent Entities", len(parent_list))
    with kpi3:
        st.metric("Avg Regional Income", f"${d_df['income'].mean():,.0f}")

    st.markdown('<div class="section-header">Competitive Landscape (Parent Company Level)</div>', unsafe_allow_html=True)
    
    # Main Dashboard Area
    layout_col1, layout_col2 = st.columns([1, 1.5])
    
    with layout_col1:
        st.subheader("Entity Analysis")
        # DROPDOWN: Guaranteed unique due to MDM Logic + .unique()
        target_entity = st.selectbox("Select Parent Company", parent_list)
        
        entity_skus = m_df[m_df['parent_company'] == target_entity]
        st.write(f"This entity controls **{len(entity_skus)} SKUs** in the current scan.")
        
        if st.button("Generate Executive Brief"):
            st.info(f"Briefing for {target_entity} would appear here based on trend PDF context.")

    with layout_col2:
        st.subheader("Share of Shelf (Top 10 Parents)")
        shelf_share = m_df['parent_company'].value_counts().head(10)
        st.bar_chart(shelf_share)

    # --- AUDIT TRAIL (For demonstrating the tech to leaders) ---
    with st.expander("🔍 AI Data Normalization Audit"):
        st.write("This table shows how messy brand data was consolidated into clean Parent Entities.")
        audit_df = m_df[['brands', 'parent_company']].drop_duplicates()
        st.dataframe(audit_df, use_container_width=True)

else:
    st.info("Please enter your API keys and click 'Run Market Scan' to begin.")
