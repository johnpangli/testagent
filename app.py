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

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    div.block-container {padding-top: 1.5rem; max-width: 1400px;}
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-size: 18px; font-weight: 600;
    }
    th { background-color: #f0f2f6; color: #1e3a8a !important; text-align: left; padding: 12px; }
    td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
    .stMetric { background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'data_fetched' not in st.session_state: st.session_state.data_fetched = False
if 'market_df' not in st.session_state: st.session_state.market_df = None
if 'demographics_df' not in st.session_state: st.session_state.demographics_df = None

# --- 3. HELPER: MDM & JSON PARSING ---
def safe_json_parse(text):
    text = re.sub(r'```json\s?|```', '', text).strip()
    try: return json.loads(text)
    except:
        try:
            text = re.sub(r',\s*\}', '}', text); text = re.sub(r',\s*\]', ']', text)
            return json.loads(text)
        except: return {}

def get_canonical_parent_map(messy_brands, api_key):
    """The MDM Specialist: One-shot consolidation of parent companies."""
    if not messy_brands or not api_key: return {}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    ACT AS: Enterprise MDM Specialist. 
    TASK: Map messy brand strings to one true Parent Company.
    RULES: 
    - Consolidate: "365", "Whole Foods", "365 Everyday Value" -> "Amazon/Whole Foods"
    - Consolidate: "Blue Diamond", "Blue Diamond Almonds" -> "Blue Diamond Growers"
    - Resolve: "Wright", "Wright Brand" -> "Tyson Foods"
    - Hierarchy: Identify corporate owners (Hormel, Kraft Heinz, etc.)
    LIST: {messy_brands}
    RETURN JSON: {{"Mapping": [{{"raw": "Messy Name", "parent": "Clean Parent"}}]}}
    """
    try:
        res = model.generate_content(prompt)
        data = safe_json_parse(res.text)
        return {item['raw']: item['parent'] for item in data.get('Mapping', [])}
    except: return {b: b for b in messy_brands}

# --- 4. DATA ENGINES ---
def fetch_data(category, region, gemini_key, census_key):
    # --- 4a. Market Data (Pagination) ---
    tech_tag = {"Bacon": "bacons", "Peanut Butter": "peanut-butters", "Snack Nuts": "nuts"}.get(category, category.lower())
    all_prods = []
    for p in range(1, 6):
        url = f"https://world.openfoodfacts.org/cgi/search.pl?action=process&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States&json=1&page_size=100&page={p}&fields=product_name,brands,ingredients_text,labels_tags"
        try:
            r = requests.get(url, timeout=15).json().get('products', [])
            if not r: break
            all_prods.extend(r)
        except: break
    
    df = pd.DataFrame(all_prods).dropna(subset=['brands'])
    df['brands'] = df['brands'].str.strip(',')
    
    # Run MDM Engine
    unique_messy = df['brands'].unique().tolist()
    parent_map = get_canonical_parent_map(unique_messy, gemini_key)
    df['parent_company'] = df['brands'].map(parent_map).fillna(df['brands'])
    
    # --- 4b. Census Data ---
    c = Census(census_key)
    states = {"Midwest": ["MI", "IL", "OH"], "Northeast": ["NY", "PA"], "South": ["TX", "FL"], "West": ["CA", "WA"]}.get(region, ["MI"])
    all_census = []
    for s in states:
        try:
            fips = us.states.lookup(s).fips
            all_census.extend(c.acs5.state_zipcode(('B01003_001E', 'B19013_001E'), fips, Census.ALL))
        except: continue
    d_df = pd.DataFrame(all_census)
    d_df['income'] = pd.to_numeric(d_df['B19013_001E'], errors='coerce')
    
    return df, d_df[d_df['income'] > 0]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🤖 Strategy Agent")
    GEMINI_API = st.text_input("Gemini Key", type="password")
    CENSUS_API = st.text_input("Census Key", type="password")
    REGION = st.selectbox("Region", ["Midwest", "Northeast", "South", "West"])
    CATEGORY = st.selectbox("Category", ["Bacon", "Peanut Butter", "Snack Nuts"])
    files = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    run = st.button("🚀 Run Full Analysis", type="primary")

# --- 6. MAIN APP ---
if run and GEMINI_API:
    st.session_state.market_df, st.session_state.demographics_df = fetch_data(CATEGORY, REGION, GEMINI_API, CENSUS_API)
    st.session_state.data_fetched = True

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df
    
    # KPIs
    st.markdown('<div class="section-header">Market Concentration Dashboard</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("SKU Volume", len(m_df))
    # THE KEY: Parent List is Clean
    parent_list = sorted(m_df['parent_company'].unique().tolist())
    c2.metric("Clean Parent Entities", len(parent_list))
    c3.metric("Avg HHI", f"${d_df['income'].mean():,.0f}")

    # Analysis Setup
    st.divider()
    target_parent = st.selectbox("Select Your Brand (Parent Level):", parent_list)
    
    if st.button("✨ Generate Strategic Directive", type="primary"):
        with st.spinner("Synthesizing MECE Occasions & Ingredient Gaps..."):
            genai.configure(api_key=GEMINI_API)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Prepare context
            my_data = m_df[m_df['parent_company'] == target_parent].head(10).to_string()
            comp_data = m_df[m_df['parent_company'] != target_parent].head(20).to_string()
            
            prompt = f"""
            ACT AS: Chief Strategy Officer. 
            CONTEXT: Analyzing '{CATEGORY}' for Parent Entity '{target_parent}'.
            
            DATA:
            - My Portfolio: {my_data}
            - Competitor Landscape: {comp_data}
            - Macro Income: ${d_df['income'].mean():,.0f}
            
            TASK: Return a JSON-only response with:
            1. "exec_summary": 2-sentence BLUF.
            2. "occasions_matrix": List of 3 objects with keys: [occasion, leader, leader_tactic, my_gap, driver]
            3. "claims_strategy": {{"leader_claims": "...", "our_gap": "..."}}
            4. "strategic_questions": List of 3 difficult assortment/architecture questions.
            5. "ingredient_audit": List of objects with [type, my_ingredients, competitor_ingredients, implication]
            """
            
            try:
                res = model.generate_content(prompt)
                result = safe_json_parse(res.text)
                
                # --- RENDER STRATEGY ---
                st.markdown("## 📋 Executive Briefing")
                st.info(result.get("exec_summary"))
                
                st.markdown("### 📊 MECE Occasion Matrix")
                occ_df = pd.DataFrame(result.get("occasions_matrix", []))
                st.table(occ_df) # Simplified table for clean UI
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🏷️ Claims Architecture")
                    claims = result.get("claims_strategy", {})
                    st.success(f"**Competitors Winning On:** {claims.get('leader_claims')}")
                    st.error(f"**Our Critical Gaps:** {claims.get('our_gap')}")
                
                with col2:
                    st.subheader("🧐 Strategic Questions")
                    for q in result.get("strategic_questions", []):
                        st.warning(f"👉 {q}")
                
                st.markdown("### 🔬 Technical Ingredient Audit")
                ing_df = pd.DataFrame(result.get("ingredient_audit", []))
                st.table(ing_df)
                
            except Exception as e:
                st.error(f"Analysis Generation Failed: {e}")

    # Audit Trail
    with st.expander("🔍 View Data Audit (Raw vs AI Canonical)"):
        st.dataframe(m_df[['brands', 'parent_company']].drop_duplicates())
