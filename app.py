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
    /* Table Styling for the AI Reports */
    table {width: 100%; border-collapse: collapse; margin: 10px 0;}
    th { background-color: #f8fafc; color: #1e3a8a; text-align: left; padding: 12px; border-bottom: 2px solid #e2e8f0; }
    td { padding: 10px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
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

# --- 3. MDM & TEXT PROCESSING ENGINES ---

def get_canonical_parent_map(messy_brands, api_key):
    """Consolidates messy brand strings into Parent Companies via Gemini."""
    if not messy_brands or not api_key: return {}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    ACT AS: Enterprise Master Data Management (MDM) Specialist.
    TASK: Map messy brand strings to their ONE true Parent Company.
    RULES: 
    1. Consolidate: "365", "Whole Foods" -> "Amazon/Whole Foods".
    2. Resolve Parents: "Wright" -> "Tyson Foods", "Jimmy Dean" -> "Tyson Foods".
    3. Return valid JSON only.
    
    LIST: {messy_brands}
    RETURN: {{"Mapping": [{{"raw": "Name", "canonical_parent": "Clean Name"}}]}}
    """
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json\s?|```', '', response.text).strip()
        data = json.loads(clean_json)
        return {item['raw']: item['canonical_parent'] for item in data.get('Mapping', [])}
    except:
        return {b: b for b in messy_brands}

def process_trends(files):
    """Extracts text from uploaded PDF trend reports."""
    if not files: return ""
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:5]])
        except: pass
    return text[:20000]

# --- 4. DATA ACQUISITION ---

REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA", "CT"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN"],
    "West": ["CA", "WA", "AZ", "CO", "OR"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", "Beef Jerky": "meat-snacks"
}

def fetch_market_intelligence(category, api_key):
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'StrategicHub/2.0'}
    all_products = []
    
    for page in range(1, 4): # 3 Pages for balanced speed/depth
        params = {
            "action": "process", "tagtype_0": "categories", "tag_contains_0": "contains",
            "tag_0": tech_tag, "json": "1", "page_size": 100, "page": page,
            "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n",
            "cc": "us"
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            products = r.json().get('products', [])
            if not products: break
            all_products.extend(products)
        except: break

    df = pd.DataFrame(all_products)
    if df.empty: return df

    df['brands'] = df['brands'].str.strip().str.strip(',').fillna("Unbranded")
    unique_messy = df['brands'].unique().tolist()
    
    # MDM Resolution
    parent_map = get_canonical_parent_map(unique_messy, api_key)
    df['parent_company'] = df['brands'].map(parent_map).fillna(df['brands'])
    return df

def fetch_demographics(api_key, region):
    if not api_key: return None
    c = Census(api_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []
    # Fetching Pop, Income, and Poverty
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    for s_code in states:
        try:
            state_obj = us.states.lookup(s_code)
            res = c.acs5.state_zipcode(vars, state_obj.fips, Census.ALL)
            all_data.extend(res)
        except: continue
        
    df = pd.DataFrame(all_data)
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    p_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    p_den = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (p_num / p_den.replace(0, 1)) * 100
    return df[df['income'] > 0]

# --- 5. SIDEBAR CONTROLS ---

with st.sidebar:
    st.header("Hub Configuration")
    GEMINI_API = st.text_input("Gemini API Key", type="password")
    CENSUS_API = st.text_input("Census API Key", type="password")
    
    st.divider()
    REGION = st.selectbox("Strategic Region", list(REGION_MAP.keys()))
    CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    
    st.subheader("Trend Context")
    uploaded_pdfs = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    
    execute = st.button("▶ Run Analysis Engine", type="primary")

# --- 6. MAIN DASHBOARD LOGIC ---

st.title("Strategic Intelligence Hub")

if execute and GEMINI_API:
    with st.status("Gathering Intelligence...", expanded=True) as status:
        st.write("🛰️ Fetching Market & Competitor Data...")
        st.session_state.market_df = fetch_market_intelligence(CATEGORY, GEMINI_API)
        st.write("📊 Pulling Census Demographics...")
        st.session_state.demographics_df = fetch_demographics(CENSUS_API, REGION)
        st.write("📄 Ingesting Trend Reports...")
        st.session_state.trends_text = process_trends(uploaded_pdfs)
        
        st.session_state.data_fetched = True
        status.update(label="Intelligence Gathered", state="complete")

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df
    
    # Top-Level KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market SKUs", len(m_df))
    parent_list = sorted(m_df['parent_company'].unique().tolist())
    k2.metric("Parent Entities", len(parent_list))
    k3.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
    k4.metric("Avg Poverty", f"{d_df['poverty_rate'].mean():.1f}%")

    st.markdown('<div class="section-header">Entity Selection & Competitive Benchmarking</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        target_entity = st.selectbox("Select Parent Company for Analysis", parent_list)
        entity_skus = m_df[m_df['parent_company'] == target_entity]
        st.write(f"Analyzed **{len(entity_skus)}** product profiles for {target_entity}.")
        
        generate_report = st.button("✨ Generate Strategic Directive")

    with col_right:
        st.subheader("Share of Shelf (Top 10 Parents)")
        shelf_share = m_df['parent_company'].value_counts().head(10)
        st.bar_chart(shelf_share)

    # --- 7. STRATEGIC DIRECTIVE GENERATION ---
    if generate_report:
        with st.spinner(f"Synthesizing Directive for {target_entity}..."):
            genai.configure(api_key=GEMINI_API)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Prepare Competitor Context (Top 3 Rivals)
            rivals = m_df[m_df['parent_company'] != target_entity]['parent_company'].value_counts().head(3).index.tolist()
            
            def get_entity_context(entity_name):
                subset = m_df[m_df['parent_company'] == entity_name].head(8)
                return "\n".join([f"- {r.product_name} | Ingredients: {str(r.ingredients_text)[:200]} | Labels: {r.labels_tags}" for _, r in subset.iterrows()])

            prompt = f"""
            ACT AS: Chief Strategy Officer.
            CONTEXT: Category: {CATEGORY} | Region: {REGION}
            TARGET ENTITY: {target_entity}
            RIVALS: {', '.join(rivals)}
            
            DEMOGRAPHICS: Avg Income ${d_df['income'].mean():,.0f}, Poverty Rate {d_df['poverty_rate'].mean():.1f}%
            TREND DATA: {st.session_state.trends_text[:10000]}
            
            PRODUCT DATA:
            MY PRODUCTS: {get_entity_context(target_entity)}
            RIVAL PRODUCTS: {chr(10).join([f"---{r}---" + get_entity_context(r) for r in rivals])}

            TASK: Return JSON ONLY with keys:
            1. "executive_summary": 2-sentence BLUF.
            2. "occasions_matrix": List of 3 objects (occasion_name, rival_leader, my_gap, strategic_attribute).
            3. "ingredient_audit": List of 3 objects for a table (type, my_brand_usage, rival_usage, implication).
            4. "strategic_questions": List of 3 difficult questions for the board.
            """

            try:
                response = model.generate_content(prompt)
                res = json.loads(re.sub(r'```json\s?|```', '', response.text).strip())
                
                st.divider()
                st.markdown("## 📋 Executive Strategic Directive")
                st.info(res.get("executive_summary"))
                
                st.subheader("📊 Strategic Occasion Matrix")
                occ_df = pd.DataFrame(res.get("occasions_matrix", []))
                st.write(occ_df.to_html(index=False), unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🔬 Technical Ingredient Audit")
                    ing_df = pd.DataFrame(res.get("ingredient_audit", []))
                    st.write(ing_df.to_html(index=False), unsafe_allow_html=True)
                with c2:
                    st.subheader("🧐 Strategic Questions")
                    for q in res.get("strategic_questions", []):
                        st.warning(f"👉 {q}")
                        
            except Exception as e:
                st.error(f"Strategic Engine Error: {e}")

    # Audit Trail
    with st.expander("🔍 AI Data Normalization Audit"):
        st.dataframe(m_df[['brands', 'parent_company']].drop_duplicates(), use_container_width=True)

else:
    st.info("Configure keys and parameters in the sidebar, then click 'Run Analysis Engine'.")
