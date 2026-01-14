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
    
    /* Professional Color Scheme */
    :root {
        --primary-blue: #1e3a8a;
        --accent-blue: #3b82f6;
        --light-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #1e293b;
    }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: var(--primary-blue);
    }
    
    /* Tables */
    table {
        width: 100%; 
        border-collapse: collapse;
        font-size: 14px;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    th {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white !important;
        text-align: left;
        padding: 12px 16px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.5px;
    }
    td { 
        padding: 12px 16px; 
        border-bottom: 1px solid var(--border-gray);
        color: var(--text-dark);
    }
    tr:last-child td {
        border-bottom: none;
    }
    tr:hover {
        background-color: var(--light-gray);
    }
    
    /* Cards */
    .insight-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid var(--accent-blue);
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .insight-card h4 {
        color: var(--primary-blue);
        margin-bottom: 8px;
        font-size: 16px;
        font-weight: 600;
    }
    
    .insight-card p {
        color: var(--text-dark);
        line-height: 1.6;
        margin: 0;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        margin: 24px 0 16px 0;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.15);
    }
    
    /* Checkboxes */
    .stCheckbox {
        padding: 8px 0;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-blue);
    }
    
    /* Remove excessive spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 6px;
        border-left-width: 4px;
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
if 'selected_brand' not in st.session_state:
    st.session_state.selected_brand = None

# --- 3. HELPER FUNCTIONS ---

def safe_json_parse(text):
    """Robust JSON parser for LLM outputs"""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r',\s*\}', '}', text)
        text = re.sub(r',\s*\]', ']', text)
        try:
            return json.loads(text)
        except:
            return {}

def ai_brand_cleaner(df, api_key):
    """Uses Gemini to consolidate messy brand names."""
    if df.empty or not api_key: 
        return df
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    unique_brands = df['brands'].dropna().unique().tolist()
    if len(unique_brands) < 2: 
        return df

    if len(unique_brands) > 60:
        unique_brands = unique_brands[:60]

    prompt = f"""
    You are a CPG Data Cleaning Expert. I have a messy list of food brands.
    Merge variations (e.g., "KIRKLAND", "Kirkland Signature" -> "Kirkland Signature").
    Merge "Wright", "Wright Brand" -> "Wright Brand".
    
    RAW LIST: {unique_brands}
    
    RETURN ONLY JSON (No markdown, No text):
    {{
        "brand_map": {{
            "Messy Name 1": "Clean Name",
            "Messy Name 2": "Clean Name"
        }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        mapping = safe_json_parse(response.text)
        b_map = mapping.get('brand_map', {})
        
        df['brand_clean'] = df['brands'].apply(lambda x: b_map.get(x, x))
        df['brand_clean'] = df['brand_clean'].astype(str).str.title().str.strip()
        return df
    except Exception as e:
        print(f"Janitor Failed: {e}")
        df['brand_clean'] = df['brands']
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
    "Bacon": "bacons", 
    "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", 
    "Beef Jerky": "meat-snacks", 
    "Coffee": "coffees", 
    "Cereal": "breakfast-cereals", 
    "Chips": "chips"
}

def get_demographics(api_key, region_input):
    """Fetch census data with FIXED case-sensitive region lookup"""
    if not api_key: 
        return None
    
    c = Census(api_key)
    
    # FIX: Use exact case from REGION_MAP keys
    state_codes = REGION_MAP.get(region_input, [region_input.upper()])
    
    # Convert state codes to state objects
    target_states = []
    for s in state_codes:
        state_obj = us.states.lookup(s)
        if state_obj:
            target_states.append(state_obj)
    
    if not target_states:
        return None
    
    all_zips = []
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    def fetch_wrapper(state):
        try: 
            return c.acs5.state_zipcode(vars, state.fips, Census.ALL)
        except Exception as e:
            print(f"Error fetching {state.abbr}: {e}")
            return []
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_wrapper, s): s for s in target_states}
        for future in as_completed(futures):
            res = future.result()
            if res: 
                all_zips.extend(res)
            
    if not all_zips: 
        return None
        
    df = pd.DataFrame(all_zips)
    df = df.rename(columns={'zip code tabulation area': 'zip_code'})
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    poverty_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    poverty_denom = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (poverty_num / poverty_denom.replace(0, 1)) * 100
    df = df[(df['income'] > 0) & (df['population'] > 1000)]
    
    return df.sort_values(['population'], ascending=False).head(20)

def get_market_data(category_input, gemini_key):
    """Fetch and clean market data from OpenFoodFacts"""
    human_category = category_input
    technical_tag = CATEGORY_MAP.get(human_category, human_category.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'StrategicIntelligenceHub/1.0 (Streamlit; +https://streamlit.io)'}

    all_products = []
    status_text = st.empty()
    
    for page in range(1, 4):
        status_text.text(f"Fetching page {page}...")
        params = {
            "action": "process", 
            "tagtype_0": "categories", 
            "tag_contains_0": "contains",
            "tag_0": technical_tag, 
            "json": "1", 
            "page_size": 100, 
            "page": page,
            "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n,last_updated_t",
            "cc": "us", 
            "sort_by": "unique_scans_n"
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            d = r.json().get('products', [])
            if not d: 
                break
            all_products.extend(d)
            time.sleep(1.0) 
        except Exception: 
            break

    if len(all_products) < 5:
        status_text.text("Switching to keyword search...")
        for page in range(1, 4):
            params = {
                "action": "process", 
                "search_terms": human_category, 
                "json": "1",
                "page_size": 100, 
                "page": page,
                "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n,last_updated_t",
                "cc": "us", 
                "sort_by": "unique_scans_n"
            }
            try:
                r = requests.get(url, params=params, headers=headers, timeout=20)
                d = r.json().get('products', [])
                if not d: 
                    break
                all_products.extend(d)
                time.sleep(1.0)
            except: 
                break
             
    status_text.empty()
    df = pd.DataFrame(all_products)
    
    if not df.empty:
        df = df.drop_duplicates(subset=['product_name'])
        if 'countries_tags' in df.columns:
            df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]
        
        df['brands'] = df['brands'].astype(str)
        df = df[~df['brands'].isin(['nan', 'None', '', 'Unknown', 'null'])]
        
        status_text.text("Cleaning brand names with AI...")
        df = ai_brand_cleaner(df, gemini_key)
        status_text.empty()
        
    return df

def process_trends(files):
    """Extract text from uploaded PDF files"""
    if not files: 
        return "No specific trend PDF files provided. Use general knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:3]])
        except: 
            pass
    return text[:15000]

# --- 5. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### Configuration Panel")
    st.caption("Strategic Intelligence Hub v3.0")
    st.markdown("---")
    
    with st.expander("🔐 API Credentials", expanded=False):
        default_gemini = st.secrets.get("GEMINI_API_KEY", "")
        default_census = st.secrets.get("CENSUS_API_KEY", "")
        GEMINI_API_KEY = st.text_input("Gemini API Key", value=default_gemini, type="password")
        CENSUS_API_KEY = st.text_input("Census API Key", value=default_census, type="password")

    st.markdown("### Analysis Parameters")
    TARGET_REGION = st.selectbox(
        "Geographic Region", 
        ["Midwest", "Northeast", "South", "West", "USA"],
        help="Select the target market region for demographic analysis"
    )
    
    TARGET_CATEGORY = st.selectbox(
        "Product Category", 
        ["Bacon", "Peanut Butter", "Snack Nuts", "Coffee", "Cereal", "Beef Jerky", "Chips"],
        help="Select the product category to analyze"
    )
    
    st.markdown("### Trend Intelligence")
    uploaded_files = st.file_uploader(
        "Upload Trend Reports (PDF)", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Optional: Upload industry trend reports for contextualized analysis"
    )
    
    st.markdown("---")
    
    if st.button("▶ Run Data Collection", type="primary", use_container_width=True):
        st.session_state.trigger_fetch = True
    else:
        st.session_state.trigger_fetch = False

# --- 6. MAIN APPLICATION ---
st.title("Strategic Intelligence Hub")
st.caption("Enterprise-Grade Market Analysis & Competitive Intelligence")

if st.session_state.trigger_fetch:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("⚠ Configuration Required: Please provide both API keys in the sidebar.")
    else:
        with st.status("Processing data collection workflow...", expanded=True) as status:
            st.write("→ Collecting demographic data from US Census Bureau...")
            st.session_state.demographics_df = get_demographics(CENSUS_API_KEY, TARGET_REGION)
            
            st.write(f"→ Gathering market intelligence for {TARGET_CATEGORY}...")
            st.session_state.market_df = get_market_data(TARGET_CATEGORY, GEMINI_API_KEY)
            
            st.write("→ Processing trend documentation...")
            st.session_state.trends_text = process_trends(uploaded_files)
            
            st.session_state.data_fetched = True
            status.update(label="✓ Data collection complete", state="complete", expanded=False)

# --- 7. ANALYSIS DASHBOARD ---
if st.session_state.data_fetched:
    if st.session_state.demographics_df is None or st.session_state.demographics_df.empty:
        st.error(f"⚠ Data Collection Failed: Unable to retrieve census data for {TARGET_REGION}. Please verify API key and region selection.")
    elif st.session_state.market_df is None or st.session_state.market_df.empty:
        st.error(f"⚠ Data Collection Failed: No market data found for {TARGET_CATEGORY}.")
    else:
        d_df = st.session_state.demographics_df
        m_df = st.session_state.market_df
        
        # --- MARKET OVERVIEW SECTION ---
        st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Household Income", 
                f"${d_df['income'].mean():,.0f}",
                help="Average median household income across target ZIP codes"
            )
        
        with col2:
            st.metric(
                "Poverty Rate", 
                f"{d_df['poverty_rate'].mean():.1f}%",
                help="Average poverty rate across target demographics"
            )
        
        with col3:
            st.metric(
                "Total Population", 
                f"{d_df['population'].sum():,.0f}",
                help="Total population in analyzed ZIP codes"
            )
        
        with col4:
            st.metric(
                "Brands Analyzed", 
                f"{m_df['brand_clean'].nunique()}",
                help="Number of unique brands in competitive set"
            )
        
        st.markdown("---")
        
        # --- BRAND SELECTION ---
        st.markdown('<div class="section-header">Competitive Analysis Setup</div>', unsafe_allow_html=True)
        
        brand_list = sorted(m_df['brand_clean'].unique().tolist())
        
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            my_brand = st.selectbox(
                "Select Your Brand", 
                brand_list,
                help="Choose your brand for competitive positioning analysis"
            )
            st.session_state.selected_brand = my_brand
        
        with col_b:
            st.markdown("**Competitive Set**")
            comp_df = m_df[m_df['brand_clean'] != my_brand]
            top_movers = comp_df.groupby('brand_clean')['unique_scans_n'].sum().sort_values(ascending=False).head(5).index.tolist() if not comp_df.empty else []
            if top_movers:
                st.info(", ".join(top_movers))
            else:
                st.warning("Insufficient competitor data")
        
        st.markdown("---")
        
        # --- ANALYSIS MODULE SELECTION ---
        st.markdown('<div class="section-header">Analysis Modules</div>', unsafe_allow_html=True)
        st.caption("Select the strategic analyses you want to generate")
        
        col_check1, col_check2 = st.columns(2)
        
        with col_check1:
            run_occasions = st.checkbox("**Strategic Occasion Matrix**", value=True, help="Identify distinct consumer occasions and competitive gaps")
            run_claims = st.checkbox("**Claims & Positioning Strategy**", value=True, help="Analyze competitive claims and positioning opportunities")
        
        with col_check2:
            run_questions = st.checkbox("**Strategic Interrogation**", value=True, help="Generate critical questions for strategy refinement")
            run_ingredients = st.checkbox("**Technical Ingredient Audit**", value=False, help="Deep-dive ingredient comparison vs. competitors")
        
        st.markdown("---")
        
        # --- GENERATE ANALYSIS ---
        if st.button("▶ Generate Strategic Analysis", type="primary", use_container_width=True):
            if not any([run_occasions, run_claims, run_questions, run_ingredients]):
                st.warning("Please select at least one analysis module.")
            else:
                with st.spinner("Generating strategic intelligence..."):
                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-2.5-pro')

                    def get_summary(b_name):
                        d = m_df[m_df['brand_clean'] == b_name].head(5)
                        summary = []
                        for _, r in d.iterrows():
                            summary.append(f"Item: {r.get('product_name','')} | Claims: {r.get('labels_tags','')} | Ing: {str(r.get('ingredients_text',''))[:150]}...")
                        return "\n".join(summary)

                    # Build dynamic prompt based on selected modules
                    modules_requested = []
                    if run_occasions:
                        modules_requested.append('"occasions_matrix": A list of 3 objects with: occasion_name, competitor_leader, competitor_tactic, my_gap, strategic_attribute')
                    if run_claims:
                        modules_requested.append('"claims_strategy": {"competitor_wins": "...", "my_gaps": "..."}')
                    if run_questions:
                        modules_requested.append('"strategic_questions": A list of 3 strings asking difficult questions')
                    if run_ingredients:
                        modules_requested.append('"ingredient_audit": A list of objects with: ingredient_type, my_brand, competitor_1, competitor_2, implication')

                    prompt = f"""
                    ACT AS: Chief Strategy Officer for a Fortune 500 CPG Company.
                    CONSTRAINTS: Data sources are imperfect (OpenFoodFacts is user-generated, trends are limited to provided PDFs).
                    
                    CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
                    
                    DATA: 
                    - Demographics: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%
                    - MY BRAND: {my_brand} (Items: {get_summary(my_brand)})
                    - COMPETITORS: {top_movers} (Items: {chr(10).join([get_summary(c) for c in top_movers])})
                    - TRENDS: {st.session_state.trends_text}
                    
                    TASK: Generate ONLY the following modules requested:
                    {chr(10).join(modules_requested)}
                    
                    ALSO ALWAYS INCLUDE:
                    "executive_summary": A 2-3 sentence executive summary.
                    
                    RETURN ONLY VALID JSON with the requested keys. No markdown, no preamble.
                    """

                    try:
                        response = model.generate_content(prompt)
                        txt = response.text.strip()
                        txt = re.sub(r'```json', '', txt, flags=re.IGNORECASE).replace('```', '').replace(',}', '}')
                        
                        try: 
                            result = json.loads(txt)
                        except: 
                            start = txt.find('{')
                            end = txt.rfind('}') + 1
                            result = json.loads(txt[start:end])

                        # --- RENDER RESULTS ---
                        st.markdown('<div class="section-header">Strategic Intelligence Report</div>', unsafe_allow_html=True)
                        
                        # Executive Summary
                        exec_summary = result.get("executive_summary", "Analysis complete.")
                        st.markdown(f'<div class="insight-card"><h4>Executive Summary</h4><p>{exec_summary}</p></div>', unsafe_allow_html=True)
                        
                        # Occasions Matrix
                        if run_occasions and "occasions_matrix" in result:
                            st.markdown("### Strategic Occasion Matrix")
                            st.caption("Mutually exclusive consumer occasions and competitive positioning")
                            occasions = result.get("occasions_matrix", [])
                            if occasions:
                                occ_df = pd.DataFrame(occasions)
                                occ_df = occ_df.rename(columns={
                                    "occasion_name": "Occasion",
                                    "strategic_attribute": "Key Driver",
                                    "competitor_leader": "Leading Competitor",
                                    "competitor_tactic": "Competitive Approach",
                                    "my_gap": "Strategic Gap"
                                })
                                st.write(occ_df.to_html(index=False, escape=False), unsafe_allow_html=True)
                            st.markdown("---")
                        
                        # Claims Strategy
                        if run_claims and "claims_strategy" in result:
                            st.markdown("### Claims & Positioning Strategy")
                            claims = result.get("claims_strategy", {})
                            
                            col_claim1, col_claim2 = st.columns(2)
                            
                            with col_claim1:
                                st.markdown(f'<div class="insight-card"><h4>Competitor Strengths</h4><p>{claims.get("competitor_wins", "N/A")}</p></div>', unsafe_allow_html=True)
                            
                            with col_claim2:
                                st.markdown(f'<div class="insight-card"><h4>Our Opportunity Gaps</h4><p>{claims.get("my_gaps", "N/A")}</p></div>', unsafe_allow_html=True)
                            
                            st.markdown("---")
                        
                        # Strategic Questions
                        if run_questions and "strategic_questions" in result:
                            st.markdown("### Strategic Interrogation")
                            st.caption("Critical questions for leadership consideration")
                            questions = result.get("strategic_questions", [])
                            for i, q in enumerate(questions, 1):
                                st.markdown(f'<div class="insight-card"><h4>Question {i}</h4><p>{q}</p></div>', unsafe_allow_html=True)
                            st.markdown("---")
                        
                        # Ingredient Audit
                        if run_ingredients and "ingredient_audit" in result:
                            st.markdown("### Technical Ingredient Audit")
                            st.caption("Comparative formulation analysis")
                            ing_audit = result.get("ingredient_audit", [])
                            if ing_audit:
                                ing_df = pd.DataFrame(ing_audit)
                                st.write(ing_df.to_html(index=False, escape=False), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Analysis generation failed: {str(e)}")
                        st.exception(e)

        # --- DATA QUALITY DISCLAIMER ---
        st.markdown("---")
        with st.expander("📋 Data Sources & Methodology", expanded=False):
            st.markdown("""
            #### Data Quality & Limitations
            
            **Market Intelligence (OpenFoodFacts)**
            - Open-source, user-generated database
            - May contain incomplete, outdated, or duplicate entries
            - AI brand consolidation applied, but treat SKU counts as directional
            - Not a substitute for syndicated POS data (Nielsen/IRI)
            
            **Demographics (US Census Bureau)**
            - Source: American Community Survey (ACS) 5-Year Estimates
            - High accuracy but lags real-time by 1-2 years
            - Reflects selected ZIP codes, not specific retail shopper profiles
            
            **Trend Context**
            - Analysis limited to uploaded PDF documents
            - If no files uploaded, relies on model's general training data
            
            **Intended Use**
            - This tool generates strategic hypotheses for further investigation
            - Always validate critical insights with primary POS/panel data
            - Recommended for opportunity identification, not final decision-making
            """)
