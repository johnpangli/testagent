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
    div.block-container {padding-top: 2rem;}
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
    table {width: 100%;}
    th {background-color: #f0f2f6; text-align: left;}
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

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("🤖 Strategy Agent")
    st.caption("v2.2 // Bulletproof Logic")
    st.markdown("---")
    
    # API Keys
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

# --- 4. BACKEND LOGIC ---
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

def get_market_data(category_input):
    human_category = category_input
    technical_tag = CATEGORY_MAP.get(human_category, human_category.lower().replace(" ", "-"))
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    all_products = []
    for page in range(1, 4):
        params = {"action": "process", "tagtype_0": "categories", "tag_contains_0": "contains", "tag_0": technical_tag, "json": "1", "page_size": 100, "page": page, "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n", "cc": "us", "sort_by": "unique_scans_n"}
        try:
            r = requests.get(url, params=params, timeout=10)
            d = r.json().get('products', [])
            if not d: break
            all_products.extend(d)
        except: break
    # Fallback Strategy
    if len(all_products) < 10:
        for page in range(1, 4):
             params = {"action": "process", "search_terms": human_category, "json": "1", "page_size": 100, "page": page, "fields": "product_name,brands,ingredients_text,labels_tags,unique_scans_n", "cc": "us"}
             try:
                r = requests.get(url, params=params, timeout=10)
                d = r.json().get('products', [])
                if not d: break
                all_products.extend(d)
             except: break
    df = pd.DataFrame(all_products)
    if not df.empty:
        # Clean brands
        df['brand_clean'] = df['brands'].astype(str).apply(lambda x: x.split(',')[0].strip())
        df = df[~df['brand_clean'].isin(['nan', 'None', '', 'Unknown'])]
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

# --- 5. EXECUTION LOGIC ---
st.title("🚀 Strategic Intelligence Hub")

# A. Trigger Data Fetch
if st.session_state.trigger_fetch:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("❌ STOP: Please configure BOTH API Keys in the sidebar.")
    else:
        with st.status("⚙️ Agent Working...", expanded=True) as status:
            st.write("📍 Triangulating Census Demographics...")
            st.session_state.demographics_df = get_demographics(CENSUS_API_KEY, TARGET_REGION)
            
            st.write(f"🛒 Scouting Market Data for '{TARGET_CATEGORY}'...")
            st.session_state.market_df = get_market_data(TARGET_CATEGORY)
            
            st.write("📄 Ingesting Trend Reports...")
            st.session_state.trends_text = process_trends(uploaded_files)
            
            st.session_state.data_fetched = True
            status.update(label="✅ Data Acquisition Complete", state="complete", expanded=False)

# B. Display Interface (With Diagnostics)
if st.session_state.data_fetched:
    # 1. DIAGNOSTIC CHECKS (Stop silent failures)
    if st.session_state.demographics_df is None or st.session_state.demographics_df.empty:
        st.error(f"❌ Census Error: No data returned. Check API Key or Region '{TARGET_REGION}'.")
    elif st.session_state.market_df is None or st.session_state.market_df.empty:
        st.error(f"❌ Market Data Error: OpenFoodFacts returned 0 items for '{TARGET_CATEGORY}'. Try a broader category.")
    else:
        # 2. RENDER DASHBOARD
        d_df = st.session_state.demographics_df
        m_df = st.session_state.market_df
        
        st.divider()
        m1, m2, m3 = st.columns([1,1,2])
        m1.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
        m2.metric("Poverty Rate", f"{d_df['poverty_rate'].mean():.1f}%")
        
        brand_list = sorted(m_df['brand_clean'].unique().tolist())
        my_brand = m3.selectbox("Select Your Brand Focus:", brand_list)

        comp_df = m_df[m_df['brand_clean'] != my_brand]
        top_movers = comp_df.groupby('brand_clean')['unique_scans_n'].sum().sort_values(ascending=False).head(5).index.tolist() if not comp_df.empty else []
        
        st.info(f"**Comparing:** {my_brand} vs {', '.join(top_movers)}")

        # 3. GENERATE STRATEGY BUTTON
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

                # --- UPDATED PROMPT: STRICTLY TACTICAL ---
                prompt = f"""
                ACT AS: Chief Strategy Officer. 
                CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
                
                DATA: 
                - Demographics: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%
                - MY BRAND: {my_brand} (Items: {get_summary(my_brand)})
                - COMPETITORS: {top_movers} (Items: {chr(10).join([get_summary(c) for c in top_movers])})
                - TRENDS: {st.session_state.trends_text}
                
                TASK: 
                First, identify the single most relevant "Emerging Occasion" (e.g., "High-Protein Breakfast").
                Then, provide a Gap Analysis and 3 specific tactical moves regarding Assortment, Distribution, and Pack Architecture.
                
                RETURN JSON ONLY with these keys:
                
                1. "executive_summary": A 2-sentence BLUF (Bottom Line Up Front) summarizing the opportunity.
                
                2. "occasion_profile": {{"name": "Occasion Name", "rationale": "Why it fits trends/region"}}
                
                3. "gap_analysis": A Markdown Table.
                   - Columns: "Attribute", "Competitor Approach", "My Brand Gap".
                   - Rows MUST include: "Pack Size/Architecture", "Distribution Channel Fit", "Flavor/Variety Assortment", "Key Claims".
                   
                4. "claims_strategy": {{"competitor_wins": "Claims they use", "my_gaps": "Claims I need"}}
                
                5. "tactical_checklist": Markdown bullet points.
                   - Pack Architecture: Compare specific sizes/formats (e.g., "Competitors use Family Packs, we need X").
                   - Assortment/Distribution: Identify missing SKUs or channel gaps (e.g., "We lack a Spicy variant which is trending").
                   - Trade/Promo: Suggest a trade strategy (e.g., "Competitors drive trial via BOGO, we should focus on X").
                   
                6. "ingredient_table": A technical Markdown table comparing specific ingredients (Oils, Sweeteners, Preservatives).
                   
                RETURN JSON ONLY. NO MARKDOWN WRAPPERS.
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

                    # --- DASHBOARD LAYOUT ---
                    st.markdown("## 📋 Executive Summary")
                    st.info(result.get("executive_summary", "No Data"))
                    
                    st.subheader(f"🎯 Target Occasion: {result.get('occasion_profile', {}).get('name', 'N/A')}")
                    st.caption(result.get('occasion_profile', {}).get('rationale', ''))
                    
                    st.divider()

                    st.subheader("📊 Strategic Gap Analysis")
                    st.markdown(result.get("gap_analysis", "No Data"))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🏷️ Claims Strategy")
                        claims = result.get("claims_strategy", {})
                        st.success(f"**Competitors Winning On:** {claims.get('competitor_wins', 'N/A')}")
                        st.error(f"**Our Critical Gaps:** {claims.get('my_gaps', 'N/A')}")
                        
                    with c2:
                        st.subheader("🛠️ Tactical Checklist (4 Ps)")
                        st.markdown(result.get("tactical_checklist", "No Data"))

                    with st.expander("🔬 Technical Ingredient Audit", expanded=False):
                        st.markdown(result.get("ingredient_table", "No Data"))

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
