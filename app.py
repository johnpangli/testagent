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

# --- 2. SESSION STATE MANAGEMENT ---
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'market_df' not in st.session_state:
    st.session_state.market_df = None
if 'demographics_df' not in st.session_state:
    st.session_state.demographics_df = None
if 'trends_text' not in st.session_state:
    st.session_state.trends_text = ""

# --- 3. THE "ULTIMATE PARENT" MDM LOGIC (LOCKED TO GEMINI-3-FLASH-PREVIEW) ---

def get_canonical_parent_map(messy_brands, api_key):
    """
    Consolidates messy brand strings into Ultimate Corporate Parents.
    """
    if not messy_brands or not api_key: return {}
    
    genai.configure(api_key=api_key)
    # MANDATORY MODEL LOCK
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
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
        # Robust JSON cleaning
        clean_json = re.sub(r'```json\s?|```', '', response.text).strip()
        data = json.loads(clean_json)
        return {item['raw']: item['canonical_parent'] for item in data.get('Mapping', [])}
    except Exception as e:
        st.error(f"MDM Engine Error: {e}")
        return {b: b for b in messy_brands}

# --- 4. DATA ACQUISITION ---

REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA"],
    "South": ["TX", "FL", "GA", "NC", "VA"],
    "West": ["CA", "WA", "AZ", "CO"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", "Beef Jerky": "meat-snacks",
    "Coffee": "coffees", "Cereal": "breakfast-cereals", "Chips": "chips"
}

def fetch_market_intelligence(category, api_key):
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    headers = {'User-Agent': 'StrategicIntelligenceHub/2.0'}
    all_products = []
    
    status_text = st.empty()
    
    for page in range(1, 6):
        status_text.text(f"🚜 Scouting Page {page} via Category Tag...")
        url = f"https://world.openfoodfacts.org/cgi/search.pl?action=process&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States&json=1&page_size=100&page={page}&fields=product_name,brands,countries_tags,ingredients_text,labels_tags,unique_scans_n"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            products = r.json().get('products', [])
            if not products: break
            all_products.extend(products)
            time.sleep(0.5)
        except: break

    status_text.empty()
    df = pd.DataFrame(all_products)
    if df.empty: return df

    df['brands'] = df['brands'].astype(str).str.strip().str.strip(',')
    df = df[~df['brands'].isin(['nan', 'None', '', 'Unknown', 'null'])]
    df = df.drop_duplicates(subset=['product_name'])
    
    # Run the "Ultimate Parent" Cleaner
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
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    for s_code in states:
        try:
            state_obj = us.states.lookup(s_code)
            res = c.acs5.state_zipcode(vars, state_obj.fips, Census.ALL)
            all_data.extend(res)
        except: continue
        
    df = pd.DataFrame(all_data)
    if df.empty: return None
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    p_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    p_den = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (p_num / p_den.replace(0, 1)) * 100
    return df[df['income'] > 0]

def process_trends(files):
    if not files: return "No trend PDFs. Use general training knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:3]])
        except: pass
    return text[:15000]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🤖 Strategy Agent")
    GEMINI_API = st.text_input("Gemini API Key", type="password")
    CENSUS_API = st.text_input("Census API Key", type="password")
    st.divider()
    TARGET_REGION = st.selectbox("Strategic Region", list(REGION_MAP.keys()))
    TARGET_CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    uploaded_files = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    execute = st.button("🚀 Run Market Scan", type="primary")

# --- 6. MAIN DASHBOARD ---
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
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Market SKUs", len(m_df))
    parent_list = sorted(m_df['parent_company'].unique().tolist())
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

    # --- THE STRATEGIC DIRECTIVE ENGINE (LOCKED TO GEMINI-3-FLASH-PREVIEW) ---
    st.divider()
    if st.button("✨ Generate Full Strategic Directive", type="primary"):
        with st.spinner("🧠 Synthesizing Strategy via Gemini-3-Flash-Preview..."):
            genai.configure(api_key=GEMINI_API)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            
            # Context Building
            comp_list = m_df[m_df['parent_company'] != my_brand]['parent_company'].value_counts().head(5).index.tolist()
            
            prompt = f"""
            ACT AS: Chief Strategy Officer.
            CONTEXT: Analyzing '{TARGET_CATEGORY}' for '{my_brand}'.
            DATA: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%.
            TOP COMPETITORS: {comp_list}.
            TRENDS: {st.session_state.trends_text}

            TASK: Return JSON ONLY (no markdown) with:
            1. "executive_summary": 2-sentence BLUF.
            2. "occasions_matrix": List of 3 objects (occasion_name, competitor_leader, competitor_tactic, my_gap, strategic_attribute).
            3. "claims_strategy": {{"competitor_wins": "text", "my_gaps": "text"}}.
            4. "strategic_questions": List of 3 difficult questions.
            5. "ingredient_audit": List of objects (ingredient_type, my_brand, competitor_1, competitor_2, implication).
            """
            
            try:
                response = model.generate_content(prompt)
                res_txt = re.sub(r'```json\s?|```', '', response.text).strip()
                result = json.loads(res_txt)

                # RENDER SECTIONS
                st.markdown("## 📋 Executive Summary")
                st.info(result.get("executive_summary"))

                st.subheader("📊 Strategic Occasion Matrix")
                st.write(pd.DataFrame(result.get("occasions_matrix")).to_html(index=False), unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🏷️ Claims Strategy")
                    st.success(f"**Competitor Wins:** {result['claims_strategy']['competitor_wins']}")
                    st.error(f"**Our Gaps:** {result['claims_strategy']['my_gaps']}")
                with c2:
                    st.subheader("🧐 Strategic Questions")
                    for q in result.get("strategic_questions", []):
                        st.warning(f"👉 {q}")

                st.divider()
                st.subheader("🔬 Technical Ingredient Audit")
                st.write(pd.DataFrame(result.get("ingredient_audit")).to_html(index=False), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis Failed: {e}")

    with st.expander("🔍 AI Normalization Audit"):
        st.dataframe(m_df[['brands', 'parent_company']].drop_duplicates())
