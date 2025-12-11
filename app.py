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
    
    # 1. Keep the original World URL (It worked for you before)
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    # 2. DEFINE HEADERS (Critical for Streamlit Cloud to not get blocked)
    headers = {
        'User-Agent': 'StrategicIntelligenceHub/1.0 (Streamlit; +https://streamlit.io)'
    }

    all_products = []
    
    # --- STRATEGY A: CATEGORY SEARCH ---
    # We use a placeholder for status to show progress in the UI
    status_text = st.empty()
    
    for page in range(1, 4):
        status_text.text(f"🚜 Scouting Page {page} via Category Tag...")
        
        params = {
            "action": "process",
            "tagtype_0": "categories",
            "tag_contains_0": "contains",
            "tag_0": technical_tag,
            "json": "1",
            "page_size": 100,
            "page": page,
            "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n",
            "cc": "us", # Keep your original Country Code logic
            "sort_by": "unique_scans_n"
        }
        
        try:
            # Increased timeout to 20s (OFF is slow) & Added Headers
            r = requests.get(url, params=params, headers=headers, timeout=20)
            d = r.json().get('products', [])
            if not d: break
            all_products.extend(d)
            
            # CRITICAL: Put the sleep back! (Prevents blocking)
            time.sleep(1.0) 
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    # --- STRATEGY B: FALLBACK KEYWORD SEARCH ---
    if len(all_products) < 5:
        status_text.text("⚠️ Low results. Switching to Keyword Search...")
        for page in range(1, 4):
             params = {
                 "action": "process",
                 "search_terms": human_category,
                 "json": "1",
                 "page_size": 100,
                 "page": page,
                 "fields": "product_name,brands,ingredients_text,labels_tags,countries_tags,unique_scans_n",
                 "cc": "us",
                 "sort_by": "unique_scans_n"
             }
             try:
                r = requests.get(url, params=params, headers=headers, timeout=20)
                d = r.json().get('products', [])
                if not d: break
                all_products.extend(d)
                time.sleep(1.0) # CRITICAL
             except: break
             
    status_text.empty() # Clear the status message
    
    df = pd.DataFrame(all_products)
    
    if not df.empty:
        # Deduplicate
        df = df.drop_duplicates(subset=['product_name'])
        
        # Double check US filtering
        if 'countries_tags' in df.columns:
            df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]

        # Clean brands
        df['brand_clean'] = df['brands'].astype(str).apply(lambda x: x.split(',')[0].strip())
        df = df[~df['brand_clean'].isin(['nan', 'None', '', 'Unknown', 'null'])]
        
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

                # --- UPDATED PROMPT: MECE OCCASIONS & SOCRATIC TACTICS ---
                prompt = f"""
                ACT AS: Chief Strategy Officer. 
                CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
                
                DATA: 
                - Demographics: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%
                - MY BRAND: {my_brand} (Items: {get_summary(my_brand)})
                - COMPETITORS: {top_movers} (Items: {chr(10).join([get_summary(c) for c in top_movers])})
                - TRENDS: {st.session_state.trends_text}
                
                TASK: 
                1. Identify 3 DISTINCT, Mutually Exclusive, Collectively Exhaustive (MECE) Consumer Occasions relevant to this category/region (e.g., "The Morning Rush" vs "The Weekend Treat" vs "The Health Pivot").
                2. For EACH occasion, analyze the gaps.
                
                RETURN JSON ONLY with these keys:
                
                1. "executive_summary": A 2-sentence BLUF.
                
                2. "occasions_matrix": A list of 3 objects. Each object must have:
                   - "occasion_name": Title.
                   - "competitor_leader": Name of the specific competitor winning this occasion.
                   - "competitor_tactic": What specifically are they doing? (e.g., "Oscar Mayer uses Family Packs").
                   - "my_gap": What specifically am I missing? (e.g., "No Family Pack option").
                   - "strategic_attribute": The key feature driving this occasion (e.g., "Convenience" or "Clean Label").
                
                3. "claims_strategy": {{"competitor_wins": "Specific claims they use", "my_gaps": "Claims I need"}}
                
                4. "tactical_questions": A list of 3 strings. 
                   - DO NOT give advice. 
                   - ASK "Hard Questions" regarding Assortment, Distribution, or Pack Architecture based on the data.
                   - Example: "Given the 15% poverty rate, why are we prioritizing Whole Foods over Dollar General?"
                   
                5. "ingredient_audit": A list of objects for a table.
                   - "ingredient_type": (e.g., "Sweetener", "Preservative").
                   - "my_brand": What I use.
                   - "competitor_1": "Name: What they use".
                   - "competitor_2": "Name: What they use".
                   - "implication": "Why this matters for the occasion".
                   
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

                    # --- DASHBOARD RENDER ---
                    
                    # 1. Executive Summary
                    st.markdown("## 📋 Executive Summary")
                    st.info(result.get("executive_summary", "No Data"))
                    
                    st.divider()

                    # 2. MECE Occasion Matrix (Custom HTML for styling)
                    st.subheader("📊 Strategic Occasion Matrix")
                    
                    occasions = result.get("occasions_matrix", [])
                    if occasions:
                        # Convert JSON to DataFrame for clean display
                        occ_df = pd.DataFrame(occasions)
                        # Rename for UI
                        occ_df = occ_df.rename(columns={
                            "occasion_name": "Occasion",
                            "strategic_attribute": "Key Driver",
                            "competitor_leader": "Winning Rival",
                            "competitor_tactic": "Rival Approach",
                            "my_gap": "My Strategic Gap"
                        })
                        st.table(occ_df) # Use st.table to avoid CSS coloring issues

                    # 3. Claims & Questions
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🏷️ Claims Strategy")
                        claims = result.get("claims_strategy", {})
                        st.success(f"**Competitors Winning On:** {claims.get('competitor_wins', 'N/A')}")
                        st.error(f"**Our Critical Gaps:** {claims.get('my_gaps', 'N/A')}")
                        
                    with c2:
                        st.subheader("❓ Tactical Interrogation")
                        st.markdown("*(Answer these before your next review)*")
                        questions = result.get("tactical_questions", [])
                        for q in questions:
                            st.warning(f"👉 {q}")

                    # 4. Technical Ingredient Audit (Specifics)
                    st.divider()
                    st.subheader("🔬 Technical Ingredient Audit")
                    st.markdown("Direct comparison of formulation strategy vs. named competitors.")
                    
                    ing_audit = result.get("ingredient_audit", [])
                    if ing_audit:
                        ing_df = pd.DataFrame(ing_audit)
                        st.table(ing_df)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
