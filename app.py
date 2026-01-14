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
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    try:
        return json.loads(text)
    except:
        try:
            text = re.sub(r',\s*\}', '}', text)
            text = re.sub(r',\s*\]', ']', text)
            return json.loads(text)
        except:
            return {}

def ai_brand_cleaner(df, api_key):
    """
    Parent Company Engine: Normalizes brand names and maps them to corporate parents.
    """
    if df.empty or not api_key: 
        return df
    
    # Pre-cleaning: Standardize strings before AI processing to reduce SKU fragmentation
    df['brands_norm'] = df['brands'].str.replace(r'\s+(Inc\.?|Corp\.?|LLC|Foods|Brand)$', '', regex=True, flags=re.IGNORECASE)
    df['brands_norm'] = df['brands_norm'].str.strip().str.title()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Only send unique normalized brands to the AI to save tokens and improve consistency
    unique_brands = df['brands_norm'].value_counts().index.tolist()
    
    prompt = f"""
    ACT AS: CPG Data Steward.
    TASK: Map these messy retail brand strings to their true Parent Company.
    
    RULES:
    1. CONSOLIDATE: "Wright", "Wright Brand", "Wright Brand Foods" -> Parent: "Tyson Foods".
    2. CONSOLIDATE: "365", "Whole Foods" -> Parent: "Amazon/Whole Foods".
    3. RETAILER BRANDS: "Great Value" -> Parent: "Walmart", "Kirkland" -> Parent: "Costco".
    4. ACCURACY: Identify the corporate owner (e.g., "Hormel", "Kraft Heinz", "Tyson", "General Mills").
    5. FALLBACK: If owner is unknown, use the Brand Name as Parent.
    
    LIST: {unique_brands[:100]}
    
    RETURN ONLY VALID JSON:
    {{
      "mapping": [
        {{"raw": "Messy Name", "parent_company": "Parent Org"}},
        ...
      ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        data = safe_json_parse(response.text)
        parent_map = {item['raw']: item['parent_company'] for item in data.get('mapping', [])}
        
        df['parent_company'] = df['brands_norm'].map(parent_map).fillna(df['brands_norm']).str.title()
        return df
    except:
        df['parent_company'] = df['brands_norm']
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
    return df

def get_market_data(category_input, gemini_key):
    human_category = category_input
    technical_tag = CATEGORY_MAP.get(human_category, human_category.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'StrategicIntelligenceHub/1.0'}

    all_products = []
    # Pagination Loop: Fetching 5 pages for ~500 SKU depth
    for page in range(1, 6):
        params = {
            "action": "process",
            "tagtype_0": "categories", "tag_contains_0": "contains", "tag_0": technical_tag,
            "tagtype_1": "countries", "tag_contains_1": "contains", "tag_1": "United States",
            "json": "1", "page_size": 100, "page": page, "cc": "us",
            "fields": "product_name,brands,countries_tags"
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            products = r.json().get('products', [])
            if not products: break
            all_products.extend(products)
            time.sleep(0.1) 
        except: break

    df = pd.DataFrame(all_products)
    if not df.empty:
        df['brands'] = df['brands'].str.strip().str.strip(',')
        df = df.dropna(subset=['brands'])
        # Multi-layer US Filter
        if 'countries_tags' in df.columns:
            df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]
        
        df = ai_brand_cleaner(df, gemini_key)
        return df
    return pd.DataFrame()

def process_trends(files):
    if not files: return "No trend PDF files provided."
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
    run_btn = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

# --- 6. MAIN APPLICATION ---
st.title("Strategic Intelligence Hub")
st.caption("Parent Company Market Concentration & Competitive Intelligence")

if run_btn:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("⚠ Configuration Required: Please provide both API keys.")
    else:
        with st.status("Aggregating Enterprise Data...", expanded=True) as status:
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
        st.markdown('<div class="section-header">Portfolio Concentration</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Median HHI (Region)", f"${d_df['income'].median():,.0f}")
        c2.metric("Total SKU Depth", f"{len(m_df)} Units")
        c3.metric("Unique Parent Entities", m_df['parent_company'].nunique())
        
        st.markdown("---")
        
        col_sel1, col_sel2 = st.columns([1, 1.5])
        with col_sel1:
            st.markdown("### Entity Selection")
            unique_parents = sorted(m_df['parent_company'].unique().tolist())
            selected_parent = st.selectbox("Target Parent Company for Analysis", unique_parents)
            
        with col_sel2:
            st.markdown(f"**Top Competitors by SKU Volume**")
            top_comp = m_df[m_df['parent_company'] != selected_parent]['parent_company'].value_counts().head(8)
            st.bar_chart(top_comp)

        if st.button("▶ Generate Executive Intelligence Report", type="primary"):
            with st.spinner("Synthesizing strategic insights..."):
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-pro')
                
                prompt = f"""
                Analyze the {TARGET_CATEGORY} category. Focus on Parent Company: {selected_parent}.
                CONTEXT:
                - Market Depth: {len(m_df)} SKUs
                - Regional Income: ${d_df['income'].mean():,.0f}
                - Macro Trends: {st.session_state.trends_text}
                
                Provide JSON:
                1. "exec_summary": (2 sentences)
                2. "strategic_gaps": (3 specific points)
                3. "risk_assessment": (Current market threats)
                """
                
                try:
                    res = model.generate_content(prompt)
                    report = safe_json_parse(res.text)
                    st.markdown('<div class="section-header">Intelligence Output</div>', unsafe_allow_html=True)
                    st.write(report.get("exec_summary"))
                    st.info(f"**Strategic Gaps vs Leaders:** {report.get('strategic_gaps')}")
                except:
                    st.error("Report generation failed.")
    else:
        st.warning("No data found for the selected category.")
