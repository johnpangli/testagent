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
    /* Global */
    div.block-container {padding-top: 1.5rem; max-width: 1400px;}
    
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --light-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #1e293b;
    }
    
    .stButton>button { 
        width: 100%; border-radius: 6px; font-weight: 600;
        padding: 0.6rem 1rem; transition: all 0.2s;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px; font-weight: 700; color: var(--primary-blue);
    }
    
    /* Custom Table Styling */
    .report-table {
        width: 100%; border-collapse: collapse; font-size: 14px;
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 10px 0;
    }
    .report-table th {
        background: #1e3a8a; color: white; padding: 12px; text-align: left;
    }
    .report-table td {
        padding: 12px; border-bottom: 1px solid #e2e8f0;
    }

    .insight-card {
        background: white; border-radius: 8px; padding: 20px;
        margin: 12px 0; border-left: 4px solid var(--accent-blue);
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-size: 18px; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE SETUP ---
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'demographics_df' not in st.session_state:
    st.session_state.demographics_df = None
if 'market_df' not in st.session_state:
    st.session_state.market_df = None
if 'trends_text' not in st.session_state:
    st.session_state.trends_text = ""

# --- 3. HELPER FUNCTIONS ---

def safe_json_parse(text):
    """Robust JSON parser for LLM outputs"""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    try:
        return json.loads(text)
    except:
        try:
            # Fallback for common LLM trailing comma errors
            text = re.sub(r',\s*\}', '}', text)
            text = re.sub(r',\s*\]', ']', text)
            return json.loads(text)
        except:
            return {}

def ai_brand_cleaner(df, api_key):
    """
    Corporate Entity Janitor: Maps messy brands to Master Brands and Parent Companies.
    """
    if df.empty or not api_key: 
        return df
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    unique_brands = df['brands'].dropna().unique().tolist()
    if len(unique_brands) < 1: return df

    prompt = f"""
    ACT AS: CPG Data Steward.
    TASK: Clean the following list of messy food brand names.
    
    RULES:
    1. COLLAPSE VARIATIONS: Merge "365", "Whole Foods", "365 Everyday Value" into "Whole Foods Market".
    2. HIERARCHY: Map brands to their Parent Company/Owner (e.g., "Planters", "Nut-rition" -> "Hormel").
    3. PRIVATE LABEL: Identify retailer brands (e.g., "Great Value" -> "Walmart").
    4. ACCURACY: If you don't know the parent company, use the Brand Name as the Parent Company.
    
    LIST: {unique_brands[:80]}
    
    RETURN ONLY VALID JSON:
    {{
      "mapping": [
        {{"raw": "Messy Name", "clean_brand": "Consolidated Brand", "parent_company": "Parent Org"}},
        ...
      ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        data = safe_json_parse(response.text)
        mapping_list = data.get('mapping', [])
        
        brand_map = {item['raw']: item['clean_brand'] for item in mapping_list}
        parent_map = {item['raw']: item['parent_company'] for item in mapping_list}
        
        df['brand_clean'] = df['brands'].map(brand_map).fillna(df['brands']).str.title()
        df['parent_company'] = df['brands'].map(parent_map).fillna(df['brand_clean']).str.title()
        
        return df
    except Exception as e:
        df['brand_clean'] = df['brands']
        df['parent_company'] = df['brands']
        return df

# --- 4. DATA FETCHING LOGIC ---
REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA", "CT", "ME", "NH", "VT", "RI"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "SC", "AL", "LA", "MS", "AR", "KY", "WV"],
    "West": ["CA", "WA", "AZ", "CO", "OR", "NV", "UT", "ID", "MT", "WY", "NM"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", "Beef Jerky": "meat-snacks", 
    "Coffee": "coffees", "Cereal": "breakfast-cereals", "Chips": "chips"
}

def get_demographics(api_key, region_input):
    if not api_key: return None
    c = Census(api_key)
    state_codes = REGION_MAP.get(region_input, ["CA"])
    target_states = [us.states.lookup(s) for s in state_codes if us.states.lookup(s)]
    
    all_zips = []
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    def fetch_wrapper(state):
        try: return c.acs5.state_zipcode(vars, state.fips, Census.ALL)
        except: return []
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_wrapper, s) for s in target_states]
        for future in as_completed(futures):
            res = future.result()
            if res: all_zips.extend(res)
            
    if not all_zips: return None
    df = pd.DataFrame(all_zips)
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    df = df[(df['income'] > 0) & (df['population'] > 1000)]
    return df.head(50)

def get_market_data(category_input, gemini_key):
    human_category = category_input
    technical_tag = CATEGORY_MAP.get(human_category, human_category.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'StrategicIntelligenceHub/1.0'}

    params = {
        "action": "process",
        "tagtype_0": "categories", "tag_contains_0": "contains", "tag_0": technical_tag,
        "tagtype_1": "countries", "tag_contains_1": "contains", "tag_1": "United States",
        "json": "1", "page_size": 100, "cc": "us",
        "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n"
    }
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        products = r.json().get('products', [])
        df = pd.DataFrame(products)
        
        if not df.empty:
            # Multi-layer US Filter
            if 'countries_tags' in df.columns:
                df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]
            
            df = df.dropna(subset=['brands'])
            df = ai_brand_cleaner(df, gemini_key)
            return df
    except:
        return pd.DataFrame()

def process_trends(files):
    if not files: return "No specific trend PDF files provided."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:3]])
        except: pass
    return text[:10000]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### Configuration Panel")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
    CENSUS_API_KEY = st.text_input("Census API Key", type="password")
    
    TARGET_REGION = st.selectbox("Geographic Region", list(REGION_MAP.keys()))
    TARGET_CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    
    uploaded_files = st.file_uploader("Upload Trend Reports (PDF)", type=['pdf'], accept_multiple_files=True)
    
    run_btn = st.button("▶ Run Data Collection", type="primary", use_container_width=True)

# --- 6. MAIN APPLICATION ---
st.title("Strategic Intelligence Hub")
st.caption("Enterprise-Grade Market Analysis & Competitive Intelligence")

if run_btn:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("⚠ Configuration Required: Please provide both API keys.")
    else:
        with st.status("Analyzing Market Landscape...", expanded=True) as status:
            st.session_state.demographics_df = get_demographics(CENSUS_API_KEY, TARGET_REGION)
            st.session_state.market_df = get_market_data(TARGET_CATEGORY, GEMINI_API_KEY)
            st.session_state.trends_text = process_trends(uploaded_files)
            st.session_state.data_fetched = True
            status.update(label="✓ Analysis Ready", state="complete")

# --- 7. DASHBOARD ---
if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df
    
    if m_df is not None and not m_df.empty:
        st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg HHI", f"${d_df['income'].mean():,.0f}")
        c2.metric("Market Sample Size", f"{len(m_df)} SKUs")
        c3.metric("Unique Brands", m_df['brand_clean'].nunique())
        c4.metric("Parent Entities", m_df['parent_company'].nunique())
        
        st.markdown("---")
        
        # Selection Logic
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            view_type = st.radio("Analysis Level", ["Parent Company", "Individual Brand"], horizontal=True)
            group_col = 'parent_company' if view_type == "Parent Company" else 'brand_clean'
            
            unique_entities = sorted(m_df[group_col].unique().tolist())
            selected_entity = st.selectbox(f"Select Your {view_type}", unique_entities)
            
        with col_sel2:
            st.markdown(f"**Top Competitors by SKU Volume ({view_type} Level)**")
            top_comp = m_df[m_df[group_col] != selected_entity][group_col].value_counts().head(5)
            st.dataframe(top_comp, use_container_width=True)

        # --- STRATEGIC GENERATION ---
        if st.button("▶ Generate Intelligence Report", type="primary"):
            with st.spinner("Synthesizing data..."):
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-pro')
                
                # Context building
                entity_data = m_df[m_df[group_col] == selected_entity].head(10).to_string()
                comp_data = m_df[m_df[group_col] != selected_entity].head(15).to_string()
                
                prompt = f"""
                Analyze the {TARGET_CATEGORY} category in the {TARGET_REGION} region.
                LEVEL: {view_type}
                TARGET: {selected_entity}
                
                DATA CONTEXT:
                - Target Data: {entity_data}
                - Competitor Landscape: {comp_data}
                - Macro Trends: {st.session_state.trends_text}
                
                Provide a JSON response with:
                1. "executive_summary": (2 sentences)
                2. "swot": {{"strengths": [], "opportunities": []}}
                3. "competitive_gap": A table-ready list of 3 items comparing {selected_entity} to market leaders.
                """
                
                try:
                    res = model.generate_content(prompt)
                    report = safe_json_parse(res.text)
                    
                    st.markdown('<div class="section-header">Strategic Intelligence Report</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-card"><h4>Summary</h4><p>{report.get("executive_summary")}</p></div>', unsafe_allow_html=True)
                    
                    # SWOT Display
                    cs1, cs2 = st.columns(2)
                    with cs1:
                        st.success("Strengths")
                        for s in report.get("swot", {}).get("strengths", []): st.write(f"• {s}")
                    with cs2:
                        st.info("Opportunities")
                        for o in report.get("swot", {}).get("opportunities", []): st.write(f"• {o}")
                        
                except Exception as e:
                    st.error("Report generation failed.")

    else:
        st.warning("No data found for the selected category.")
