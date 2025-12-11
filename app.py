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
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE SETUP (The Fix) ---
# This keeps data alive across re-runs
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
    st.caption("v2.1 // Stable Edition")
    st.markdown("---")
    
    # API Keys (Secure)
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
    
    # Logic: When clicked, we trigger the fetch
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

# Trigger Data Fetch
if st.session_state.trigger_fetch:
    if not GEMINI_API_KEY or not CENSUS_API_KEY:
        st.error("Please configure API Keys in the sidebar.")
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

# Display Interface (Only if data exists in state)
if st.session_state.data_fetched and st.session_state.demographics_df is not None and not st.session_state.market_df.empty:
    
    # Variables from state
    d_df = st.session_state.demographics_df
    m_df = st.session_state.market_df
    
    st.divider()
    m1, m2, m3 = st.columns([1,1,2])
    m1.metric("Avg Income", f"${d_df['income'].mean():,.0f}")
    m2.metric("Poverty Rate", f"{d_df['poverty_rate'].mean():.1f}%")
    
    brand_list = sorted(m_df['brand_clean'].unique().tolist())
    
    # This selection caused the crash before. Now it won't because data_fetched is True.
    my_brand = m3.selectbox("Select Your Brand Focus:", brand_list)

    comp_df = m_df[m_df['brand_clean'] != my_brand]
    top_movers = comp_df.groupby('brand_clean')['unique_scans_n'].sum().sort_values(ascending=False).head(5).index.tolist() if not comp_df.empty else []
    
    st.info(f"**Comparing:** {my_brand} vs {', '.join(top_movers)}")

    if st.button("✨ Generate Strategic Directive", type="primary"):
        with st.spinner("🧠 Synthesizing Strategy..."):
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')

            def get_summary(b_name):
                d = m_df[m_df['brand_clean'] == b_name].head(5)
                summary = []
                for _, r in d.iterrows():
                        summary.append(f"Item: {r.get('product_name','')} | Ing: {str(r.get('ingredients_text',''))[:100]}... | Claims: {r.get('labels_tags','')}")
                return "\n".join(summary)

            prompt = f"""
            ACT AS: Chief Strategy Officer. CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
            DATA: Income ${d_df['income'].mean():,.0f}. MY BRAND: {my_brand}. COMPETITORS: {top_movers}.
            TRENDS: {st.session_state.trends_text}.
            TASK: Return a valid JSON object (NO markdown formatting) with these 3 keys:
            "gap_analysis" (Markdown table), "tactical_checklist" (3 bullet points), "moonshot" (1 sentence).
            """

            try:
                response = model.generate_content(prompt)
                txt = response.text.replace("```json", "").replace("```", "").strip()
                txt = re.sub(r',\s*\}', '}', txt)
                result = json.loads(txt)

                st.markdown("## 📊 Strategic Directive")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("1. Gap Analysis")
                    st.markdown(result.get("gap_analysis", "No Data"))
                with col2:
                    st.subheader("2. Tactics")
                    st.markdown(result.get("tactical_checklist", "No Data"))
                with col3:
                    st.subheader("3. Moonshot")
                    st.info(result.get("moonshot", "No Data"))

            except Exception as e:
                st.error(f"Error: {e}")
elif st.session_state.data_fetched:
    st.warning("⚠️ No data found. Try a different Category or Region.")


