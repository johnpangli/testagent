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

# --- 1. UI CONFIGURATION & CSS FIXES ---
st.set_page_config(page_title="Strategic Intelligence Hub", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    div.block-container {padding-top: 2rem;}
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
    
    /* --- TABLE STYLING FIX --- */
    table {width: 100%; border-collapse: collapse;}
    th {
        background-color: #f0f2f6; 
        color: #000000 !important; /* Forces black text for readability */
        text-align: left;
        padding: 10px;
        font-weight: bold;
    }
    td { padding: 8px; border-bottom: 1px solid #ddd; }
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

# --- 3. HELPER FUNCTIONS (AI JANITOR) ---

def safe_json_parse(text):
    """Robust JSON parser for LLM outputs"""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Aggressive cleanup
        text = re.sub(r',\s*\}', '}', text)
        text = re.sub(r',\s*\]', ']', text)
        try:
            return json.loads(text)
        except:
            return {}

def ai_brand_cleaner(df, api_key):
    """Uses Gemini to consolidate messy brand names."""
    if df.empty or not api_key: return df
    
    # Configure local instance for this function
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') # Use flash for speed
    
    unique_brands = df['brands'].dropna().unique().tolist()
    if len(unique_brands) < 2: return df

    # We process in chunks if there are too many brands to avoid token limits
    # But for this demo, we'll try to do it in one pass or limit to top 50
    if len(unique_brands) > 60:
        unique_brands = unique_brands[:60] # Optimization for demo speed

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
        
        # Apply mapping
        df['brand_clean'] = df['brands'].apply(lambda x: b_map.get(x, x))
        
        # Final fallback cleanup for things the LLM might have missed or if list was truncated
        df['brand_clean'] = df['brand_clean'].astype(str).str.title().str.strip()
        return df
    except Exception as e:
        print(f"Janitor Failed: {e}")
        df['brand_clean'] = df['brands'] # Fallback
        return df

# --- 4. DATA FETCHING LOGIC ---
REGION_MAP = {
    "MIDWEST": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "NORTHEAST": ["NY", "PA", "NJ", "MA"],
    "SOUTH": ["TX", "FL", "GA", "NC", "VA"],
    "WEST": ["CA", "WA", "AZ", "CO"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]
}
CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", "Snack Nuts": "nuts", 
    "Beef Jerky": "meat-snacks", "Coffee": "coffees", "Cereal": "breakfast-cereals", "Chips": "chips"
}

def get_demographics(api_key, region_input):
    if not api_key: return None
    c = Census(api_key)
    upper_input = region_input.upper()
    target_states = [us.states.lookup(s) for s in REGION_MAP.get(upper_input, [region_input])]
    target_states = [s for s in target_states if s]
    all_zips = []
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    def fetch_wrapper(s):
        try: return c.acs5.state_zipcode(vars, s.fips, Census.ALL)
        except: return []
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_wrapper, s): s for s in target_states}
        for future in as_completed(futures):
            res = future.result()
            if res: all_zips.extend(res)
            
    if not all_zips: return None
    df = pd.DataFrame(all_zips)
    df = df.rename(columns={'zip code tabulation area': 'zip_code'})
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    poverty_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    poverty_denom = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (poverty_num / poverty_denom.replace(0, 1)) * 100
    df = df[(df['income'] > 0) & (df['population'] > 1000)]
    return df.sort_values(['population'], ascending=False).head(15)

def get_market_data(category_input, gemini_key):
    human_category = category_input
    technical_tag = CATEGORY_MAP.get(human_category, human_category.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'StrategicIntelligenceHub/1.0 (Streamlit; +https://streamlit.io)'}

    all_products = []
    status_text = st.empty()
    
    # Fetch Loop
    for page in range(1, 4):
        status_text.text(f"🚜 Scouting Page {page} via Category Tag...")
        params = {
            "action": "process", "tagtype_0": "categories", "tag_contains_0": "contains",
            "tag_0": technical_tag, "json": "1", "page_size": 100, "page": page,
            "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n,last_updated_t",
            "cc": "us", "sort_by": "unique_scans_n"
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            d = r.json().get('products', [])
            if not d: break
            all_products.extend(d)
            time.sleep(1.0) 
        except Exception: break

    # Fallback Loop
    if len(all_products) < 5:
        status_text.text("⚠️ Low results. Switching to Keyword Search...")
        for page in range(1, 4):
             params = {
                 "action": "process", "search_terms": human_category, "json": "1",
                 "page_size": 100, "page": page,
                 "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n,last_updated_t",
                 "cc": "us", "sort_by": "unique_scans_n"
             }
             try:
                r = requests.get(url, params=params, headers=headers, timeout=20)
                d = r.json().get('products', [])
                if not d: break
                all_products.extend(d)
                time.sleep(1.0)
             except: break
             
    status_text.empty()
    df = pd.DataFrame(all_products)
    
    if not df.empty:
        df = df.drop_duplicates(subset=['product_name'])
        if 'countries_tags' in df.columns:
            df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]
        
        # Clean basic empty brands first
        df['brands'] = df['brands'].astype(str)
        df = df[~df['brands'].isin(['nan', 'None', '', 'Unknown', 'null'])]
        
        # --- CALL AI JANITOR HERE ---
        status_text.text("🧹 AI Janitor: Consolidating duplicate brands...")
        df = ai_brand_cleaner(df, gemini_key)
        status_text.empty()
        
    return df

def process_trends(files):
    if not files: return "No specific trend PDF files provided. Use general knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:3]])
        except: pass
    return text[:15000]

# --- 5. SIDEBAR & EXECUTION ---
with st.sidebar:
    st.title("🤖 Strategy Agent")
    st.caption("v2.3 // AI-Cleaned Data")
    st.markdown("---")
    
    with st.expander("🔑 API Keys", expanded=False):
        default_gemini = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
        default_census = st.secrets["CENSUS_API_KEY"] if "CENSUS_API_KEY" in st.secrets else ""
        GEMINI_API_KEY = st.text_input("Gemini Key", value=default_gemini, type="password")
        CENSUS_API_KEY = st.text_input("Census Key", value=default_census, type="password")

    st.subheader("🎯 Parameters")
    TARGET_REGION = st.selectbox("Target Region", ["Midwest", "Northeast", "South", "West", "USA"])
    TARGET_CATEGORY = st.selectbox("Target Category", ["Bacon", "Peanut Butter", "Snack Nuts", "Coffee", "Cereal", "Beef Jerky", "Chips"])
    
    st.subheader("📂 Trend Context")
    uploaded_files = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    
    if st.button("🚀 Run Analysis Engine", type="primary"):
        st.session_state.trigger_fetch = True
    else:
        st.session_state.trigger_fetch = False

# --- MAIN LOGIC ---
st.title("🚀 Strategic Intelligence Hub")

if st.session_state.trigger_fetch:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("❌ STOP: Please configure BOTH API Keys in the sidebar.")
    else:
        with st.status("⚙️ Agent Working...", expanded=True) as status:
            st.write("📍 Triangulating Census Demographics...")
            st.session_state.demographics_df = get_demographics(CENSUS_API_KEY, TARGET_REGION)
            
            st.write(f"🛒 Scouting & Cleaning Market Data for '{TARGET_CATEGORY}'...")
            st.session_state.market_df = get_market_data(TARGET_CATEGORY, GEMINI_API_KEY)
            
            st.write("📄 Ingesting Trend Reports...")
            st.session_state.trends_text = process_trends(uploaded_files)
            
            st.session_state.data_fetched = True
            status.update(label="✅ Data Acquisition Complete", state="complete", expanded=False)

if st.session_state.data_fetched:
    if st.session_state.demographics_df is None or st.session_state.demographics_df.empty:
        st.error(f"❌ Census Error: No data returned.")
    elif st.session_state.market_df is None or st.session_state.market_df.empty:
        st.error(f"❌ Market Data Error: No items found for '{TARGET_CATEGORY}'.")
    else:
        d_df = st.session_state.demographics_df
        m_df = st.session_state.market_df
        
        st.divider()
        m1, m2, m3 = st.columns([1,1,2])
        m1.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
        m2.metric("Poverty Rate", f"{d_df['poverty_rate'].mean():.1f}%")
        
        # --- NEW: Use Cleaned Brands ---
        brand_list = sorted(m_df['brand_clean'].unique().tolist())
        my_brand = m3.selectbox("Select Your Brand Focus:", brand_list)

        comp_df = m_df[m_df['brand_clean'] != my_brand]
        top_movers = comp_df.groupby('brand_clean')['unique_scans_n'].sum().sort_values(ascending=False).head(5).index.tolist() if not comp_df.empty else []
        
        st.info(f"**Comparing:** {my_brand} vs {', '.join(top_movers)}")

        if st.button("✨ Generate Strategic Directive", type="primary"):
            with st.spinner("🧠 Synthesizing Strategy..."):
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-2.5-pro')

                def get_summary(b_name):
                    d = m_df[m_df['brand_clean'] == b_name].head(5)
                    summary = []
                    for _, r in d.iterrows():
                          summary.append(f"Item: {r.get('product_name','')} | Claims: {r.get('labels_tags','')} | Ing: {str(r.get('ingredients_text',''))[:150]}...")
                    return "\n".join(summary)

                # --- UPDATED PROMPT: LESS AGGRESSIVE TITLES, MORE CONTEXT AWARENESS ---
                prompt = f"""
                ACT AS: Chief Strategy Officer. 
                CONSTRAINTS: You are aware that the data sources are imperfect (OpenFoodFacts is user-generated, Trends are limited to provided PDFs).
                
                CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
                
                DATA: 
                - Demographics: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%
                - MY BRAND: {my_brand} (Items: {get_summary(my_brand)})
                - COMPETITORS: {top_movers} (Items: {chr(10).join([get_summary(c) for c in top_movers])})
                - TRENDS: {st.session_state.trends_text}
                
                TASK: 
                1. Identify 3 DISTINCT, Mutually Exclusive, Collectively Exhaustive (MECE) Consumer Occasions.
                2. For EACH occasion, analyze the gaps.
                
                RETURN JSON ONLY with these keys:
                
                1. "executive_summary": A 2-sentence BLUF.
                
                2. "occasions_matrix": A list of 3 objects. Each object must have:
                   - "occasion_name": Title.
                   - "competitor_leader": Name of the specific competitor winning this occasion.
                   - "competitor_tactic": What specifically are they doing?
                   - "my_gap": What specifically am I missing?
                   - "strategic_attribute": The key feature driving this occasion.
                
                3. "claims_strategy": {{"competitor_wins": "Specific claims they use", "my_gaps": "Claims I need"}}
                
                4. "strategic_questions": A list of 3 strings. 
                   - RENAME "Tactical Interrogation" to "Strategic Questions".
                   - Ask difficult questions about Assortment or Pack Architecture based on the data.
                   
                5. "ingredient_audit": A list of objects for a table.
                   - "ingredient_type": (e.g., "Sweetener", "Preservative").
                   - "my_brand": What I use.
                   - "competitor_1": "Name: What they use".
                   - "competitor_2": "Name: What they use".
                   - "implication": "Why this matters".
                   
                RETURN JSON ONLY.
                """

                try:
                    response = model.generate_content(prompt)
                    txt = response.text.strip()
                    txt = re.sub(r'```json', '', txt, flags=re.IGNORECASE).replace('```', '').replace(',}', '}')
                    
                    try: result = json.loads(txt)
                    except: 
                        start = txt.find('{')
                        end = txt.rfind('}') + 1
                        result = json.loads(txt[start:end])

                    # --- DASHBOARD RENDER ---
                    
                    st.markdown("## 📋 Executive Summary")
                    st.info(result.get("executive_summary", "No Data"))
                    st.divider()

                    st.subheader("📊 Strategic Occasion Matrix")
                    occasions = result.get("occasions_matrix", [])
                    if occasions:
                        occ_df = pd.DataFrame(occasions)
                        occ_df = occ_df.rename(columns={
                            "occasion_name": "Occasion", "strategic_attribute": "Key Driver",
                            "competitor_leader": "Winning Rival", "competitor_tactic": "Rival Approach",
                            "my_gap": "My Strategic Gap"
                        })
                        st.write(occ_df.to_html(index=False), unsafe_allow_html=True) # HTML render for better CSS control

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🏷️ Claims Strategy")
                        claims = result.get("claims_strategy", {})
                        st.success(f"**Competitors Winning On:** {claims.get('competitor_wins', 'N/A')}")
                        st.error(f"**Our Critical Gaps:** {claims.get('my_gaps', 'N/A')}")
                        
                    with c2:
                        st.subheader("🧐 Strategic Questions") # Renamed from Tactical Interrogation
                        questions = result.get("strategic_questions", [])
                        for q in questions:
                            st.warning(f"👉 {q}")

                    st.divider()
                    st.subheader("🔬 Technical Ingredient Audit")
                    ing_audit = result.get("ingredient_audit", [])
                    if ing_audit:
                        ing_df = pd.DataFrame(ing_audit)
                        st.write(ing_df.to_html(index=False), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

        # --- DATA REALITY CHECK (DISCLAIMER SECTION) ---
        st.markdown("---")
        with st.expander("🛡️ Data Reality Check & Methodology (Read Me)", expanded=False):
            st.markdown("""
            **About the Data Sources & Limitations:**
            
            1.  **Market Data (OpenFoodFacts):** This is an open-source, user-generated database (like Wikipedia for food). 
                * *Risk:* Data may be incomplete, out of date, or contain duplicate user entries. 
                * *Mitigation:* We run an **AI Janitor** process to merge brand names, but SKU counts should be treated as directional proxies, not absolute Nielsen/IRI numbers.
            
            2.  **Demographics (US Census):** Data is sourced from the ACS 5-Year Estimates.
                * *Risk:* This data is highly accurate but trails real-time shifts by 1-2 years. It reflects the chosen Zip Codes, not specific shoppers in a specific store.
            
            3.  **Trend Context:** Analysis is limited strictly to the PDF files you upload.
                * *Risk:* If no files are uploaded, the AI relies on its general training data (cutoff dates apply).
                
            *This tool is a Hypothesis Generator, not a Validation Engine. Always verify critical insights with POS data.*
            """)
