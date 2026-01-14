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
        width: 100%; border-radius: 6px; font-weight: 600;
        padding: 0.6rem 1rem; transition: all 0.2s;
    }
    .section-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-size: 18px; font-weight: 600;
    }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
if 'trigger_fetch' not in st.session_state:
    st.session_state.trigger_fetch = False

# --- 3. MDM RESOLUTION ENGINE ---

def get_canonical_parent_map(messy_brands, api_key):
    """
    ONE-SHOT RESOLUTION: Sends unique messy strings to Gemini to map them 
    to a single Parent Company (e.g., Blue Diamond variants -> Blue Diamond Growers).
    """
    if not messy_brands or not api_key: return {}
    
    genai.configure(api_key=api_key)
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
    
    RETURN ONLY VALID JSON OBJECT:
    {{
      "Mapping": [
        {{"raw": "Messy Name", "canonical_parent": "Clean Parent Company"}},
        ...
      ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        clean_json = re.sub(r'```json\s?|```', '', response.text).strip()
        data = json.loads(clean_json)
        return {item['raw']: item['canonical_parent'] for item in data.get('Mapping', [])}
    except Exception as e:
        st.error(f"MDM Engine Error: {e}")
        return {b: b for b in messy_brands}

# --- 4. DATA ACQUISITION ---

REGION_MAP = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO"],
    "Northeast": ["NY", "PA", "NJ", "MA", "CT", "ME", "NH", "VT", "RI"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "SC", "AL", "LA", "MS", "AR", "KY", "WV"],
    "West": ["CA", "WA", "AZ", "CO", "OR", "NV", "UT", "ID", "MT", "WY", "NM"],
    "USA": ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]
}

CATEGORY_MAP = {
    "Bacon": "bacons", "Peanut Butter": "peanut-butters", 
    "Snack Nuts": "nuts", "Beef Jerky": "meat-snacks",
    "Coffee": "coffees", "Cereal": "breakfast-cereals", "Chips": "chips"
}

def fetch_market_intelligence(category, api_key):
    """Enhanced market data fetching with MDM resolution"""
    tech_tag = CATEGORY_MAP.get(category, category.lower())
    headers = {'User-Agent': 'StrategicIntelligenceHub/2.0'}
    all_products = []
    
    status_text = st.empty()
    
    # 8-Page Pagination for Depth (~800 SKUs)
    for page in range(1, 9):
        status_text.text(f"🚜 Scouting Page {page} via Category Tag...")
        url = f"https://world.openfoodfacts.org/cgi/search.pl?action=process&tagtype_0=categories&tag_contains_0=contains&tag_0={tech_tag}&tagtype_1=countries&tag_contains_1=contains&tag_1=United%20States&json=1&page_size=100&page={page}&fields=product_name,brands,countries_tags,ingredients_text,labels_tags,unique_scans_n,last_updated_t"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            products = r.json().get('products', [])
            if not products: break
            all_products.extend(products)
            time.sleep(0.1)
        except: break

    status_text.empty()
    df = pd.DataFrame(all_products)
    if df.empty: return df

    # Basic Data Scrubbing
    df['brands'] = df['brands'].astype(str).str.strip().str.strip(',')
    df = df[~df['brands'].isin(['nan', 'None', '', 'Unknown', 'null'])]
    df = df.drop_duplicates(subset=['product_name'])
    
    # US filtering
    if 'countries_tags' in df.columns:
        df = df[df['countries_tags'].astype(str).str.contains('en:united-states|us', case=False, na=False)]
    
    # Trigger One-Shot MDM Resolution
    unique_messy = df['brands'].unique().tolist()
    with st.spinner(f"🧹 AI Entity Resolution: Consolidating {len(unique_messy)} brands..."):
        parent_map = get_canonical_parent_map(unique_messy, api_key)
    
    df['parent_company'] = df['brands'].map(parent_map).fillna(df['brands'])
    
    # Classify as Private Label vs Branded
    private_label_keywords = [
        'walmart', 'great value', 'kroger', 'costco', 'kirkland', 'amazon', 
        'whole foods', '365', 'trader joe', 'target', 'good & gather', 
        'aldi', 'safeway', 'albertsons', 'wegmans', 'publix', 'heb', 
        'sam\'s choice', 'member\'s mark', 'marketside', 'simple truth'
    ]
    
    df['is_private_label'] = df['parent_company'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in private_label_keywords)
    )
    df['brand_type'] = df['is_private_label'].apply(lambda x: 'Private Label' if x else 'Branded')
    
    return df

def fetch_demographics(api_key, region):
    """Enhanced demographic fetching with poverty rates"""
    if not api_key: return None
    c = Census(api_key)
    states = REGION_MAP.get(region, ["MI"])
    all_data = []
    
    vars = ('B01003_001E', 'B19013_001E', 'B17001_002E', 'B17001_001E')
    
    def fetch_wrapper(s_code):
        try:
            state_obj = us.states.lookup(s_code)
            return c.acs5.state_zipcode(vars, state_obj.fips, Census.ALL)
        except:
            return []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_wrapper, s): s for s in states}
        for future in as_completed(futures):
            res = future.result()
            if res: all_data.extend(res)
        
    if not all_data: return None
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'zip code tabulation area': 'zip_code'})
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    df['income'] = pd.to_numeric(df['B19013_001E'], errors='coerce')
    poverty_num = pd.to_numeric(df['B17001_002E'], errors='coerce')
    poverty_denom = pd.to_numeric(df['B17001_001E'], errors='coerce')
    df['poverty_rate'] = (poverty_num / poverty_denom.replace(0, 1)) * 100
    df = df[(df['income'] > 0) & (df['population'] > 1000)]
    return df.sort_values(['population'], ascending=False).head(15)

def process_trends(files):
    """Extract text from uploaded PDF trend reports"""
    if not files: return "No specific trend PDF files provided. Use general knowledge."
    text = ""
    for f in files:
        try:
            reader = PdfReader(f)
            text += "".join([page.extract_text() for page in reader.pages[:3]])
        except: pass
    return text[:15000]

# --- 5. SIDEBAR CONTROLS ---

with st.sidebar:
    st.title("🤖 Strategy Agent")
    st.caption("v3.0 // MDM + Strategic Insights")
    st.markdown("---")
    
    with st.expander("🔑 API Keys", expanded=False):
        default_gemini = st.secrets.get("GEMINI_API_KEY", "")
        default_census = st.secrets.get("CENSUS_API_KEY", "")
        GEMINI_API = st.text_input("Gemini API Key", value=default_gemini, type="password")
        CENSUS_API = st.text_input("Census API Key", value=default_census, type="password")
    
    st.subheader("🎯 Parameters")
    TARGET_REGION = st.selectbox("Target Region", list(REGION_MAP.keys()))
    TARGET_CATEGORY = st.selectbox("Product Category", list(CATEGORY_MAP.keys()))
    
    st.subheader("📂 Trend Context")
    uploaded_files = st.file_uploader("Upload Trend PDFs", type=['pdf'], accept_multiple_files=True)
    
    st.divider()
    if st.button("🚀 Run Market Scan", type="primary"):
        st.session_state.trigger_fetch = True
    else:
        st.session_state.trigger_fetch = False

# --- 6. DASHBOARD LAYOUT ---

st.title("🚀 Strategic Intelligence Hub")
st.caption("Enterprise-Scale Parent Company Concentration & Strategic Insights")

if st.session_state.trigger_fetch:
    if not GEMINI_API or not CENSUS_API:
        st.error("❌ STOP: Please configure BOTH API Keys in the sidebar.")
    else:
        with st.status("⚙️ Agent Working...", expanded=True) as status:
            st.write("📍 Triangulating Census Demographics...")
            st.session_state.demographics_df = fetch_demographics(CENSUS_API, TARGET_REGION)
            
            st.write(f"🛒 Scouting & Cleaning Market Data for '{TARGET_CATEGORY}'...")
            st.session_state.market_df = fetch_market_intelligence(TARGET_CATEGORY, GEMINI_API)
            
            st.write("📄 Ingesting Trend Reports...")
            st.session_state.trends_text = process_trends(uploaded_files)
            
            st.session_state.data_fetched = True
            status.update(label="✅ Data Acquisition Complete", state="complete")

if st.session_state.data_fetched:
    m_df = st.session_state.market_df
    d_df = st.session_state.demographics_df
    
    if d_df is None or d_df.empty:
        st.error(f"❌ Census Error: No data returned for {TARGET_REGION}.")
    elif m_df is None or m_df.empty:
        st.error(f"❌ Market Data Error: No items found for '{TARGET_CATEGORY}'.")
    else:
        # Top-Level KPIs
        st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Market SKUs", len(m_df))
        with kpi2:
            parent_list = sorted(m_df['parent_company'].unique().tolist())
            st.metric("Clean Parent Entities", len(parent_list))
        with kpi3:
            st.metric("Avg Regional Income", f"${d_df['income'].mean():,.0f}")
        with kpi4:
            st.metric("Poverty Rate", f"{d_df['poverty_rate'].mean():.1f}%")

        st.divider()
        
        # Main Dashboard Area
        st.markdown('<div class="section-header">Competitive Landscape Analysis</div>', unsafe_allow_html=True)
        
        layout_col1, layout_col2 = st.columns([1, 1.5])
        
        with layout_col1:
            st.subheader("Entity Analysis")
            # DROPDOWN: Guaranteed unique due to MDM Logic
            my_brand = st.selectbox("Select Your Brand Focus:", parent_list)
            
            entity_skus = m_df[m_df['parent_company'] == my_brand]
            my_brand_type = entity_skus['brand_type'].iloc[0] if not entity_skus.empty else "Unknown"
            
            st.write(f"**Type:** {my_brand_type}")
            st.write(f"This entity controls **{len(entity_skus)} SKUs** in the current scan.")
            
            # Calculate top 8 competitors (excluding selected brand)
            comp_df = m_df[m_df['parent_company'] != my_brand]
            if 'unique_scans_n' in comp_df.columns:
                top_movers = comp_df.groupby('parent_company')['unique_scans_n'].sum().sort_values(ascending=False).head(8).index.tolist()
            else:
                top_movers = comp_df['parent_company'].value_counts().head(8).index.tolist()
            
            # Separate branded vs private label competitors
            top_branded = [b for b in top_movers if not m_df[m_df['parent_company'] == b]['is_private_label'].iloc[0]]
            top_private = [b for b in top_movers if m_df[m_df['parent_company'] == b]['is_private_label'].iloc[0]]
            
            if top_branded:
                st.info(f"**Top Branded Competitors:** {', '.join(top_branded[:5])}")
            if top_private:
                st.warning(f"**Private Label Competitors:** {', '.join(top_private[:3])}")

        with layout_col2:
            st.subheader("Share of Shelf (Top 10 Parents)")
            shelf_share = m_df['parent_company'].value_counts().head(10)
            st.bar_chart(shelf_share)

        # --- STRATEGIC INSIGHTS GENERATION ---
        st.divider()
        st.markdown('<div class="section-header">Strategic Intelligence</div>', unsafe_allow_html=True)
        
        if st.button("✨ Generate Strategic Directive", type="primary"):
            with st.spinner("🧠 Synthesizing Strategy..."):
                genai.configure(api_key=GEMINI_API)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')

                def get_summary(b_name):
                    d = m_df[m_df['parent_company'] == b_name].head(5)
                    brand_type = d['brand_type'].iloc[0] if not d.empty else "Unknown"
                    summary = [f"TYPE: {brand_type}"]
                    for _, r in d.iterrows():
                        summary.append(f"Item: {r.get('product_name','')} | Claims: {r.get('labels_tags','')} | Ing: {str(r.get('ingredients_text',''))[:150]}...")
                    return "\n".join(summary)

                # Separate top 8 competitors into branded and private label
                comp_summaries = []
                for comp in top_movers[:8]:
                    comp_summaries.append(f"\n{comp}:\n{get_summary(comp)}")

                prompt = f"""
                ACT AS: Chief Strategy Officer for a CPG Brand. 
                CONSTRAINTS: You are aware that the data sources are imperfect (OpenFoodFacts is user-generated, Trends are limited to provided PDFs).
                
                CONTEXT: Analyzing '{TARGET_CATEGORY}' in '{TARGET_REGION}'.
                
                DATA: 
                - Demographics: Income ${d_df['income'].mean():,.0f}, Poverty {d_df['poverty_rate'].mean():.1f}%
                - MY BRAND: {my_brand} ({my_brand_type})
                  {get_summary(my_brand)}
                
                - TOP 8 COMPETITORS (Mix of Branded & Private Label):
                  {chr(10).join(comp_summaries)}
                
                - TRENDS: {st.session_state.trends_text}
                
                CRITICAL INSTRUCTION: When analyzing competitors, DISTINGUISH between:
                1. BRANDED competitors (Blue Diamond, Wonderful, Hormel, etc.) - focus on innovation, claims, ingredients
                2. PRIVATE LABEL competitors (Walmart, Kroger, Costco/Kirkland) - focus on price positioning, "good enough" quality
                
                TASK: 
                1. Identify 3 DISTINCT, Mutually Exclusive, Collectively Exhaustive (MECE) Consumer Occasions.
                2. For EACH occasion, analyze gaps against BOTH branded and private label where relevant.
                
                RETURN JSON ONLY with these keys:
                
                1. "executive_summary": A 2-3 sentence BLUF that acknowledges the competitive set includes both branded players and private label.
                
                2. "occasions_matrix": A list of 3 objects. Each object must have:
                   - "occasion_name": Title.
                   - "competitor_leader": Name of the specific competitor winning this occasion (specify if branded or private label).
                   - "competitor_tactic": What specifically are they doing? (If private label, mention price/value strategy).
                   - "my_gap": What specifically am I missing?
                   - "strategic_attribute": The key feature driving this occasion.
                
                3. "claims_strategy": {{
                     "branded_competitor_wins": "Specific claims BRANDED competitors use",
                     "private_label_approach": "How private label positions (if relevant)",
                     "my_gaps": "Claims/positioning I need"
                   }}
                
                4. "strategic_questions": A list of 3 strings asking difficult questions about:
                   - How to compete against private label value perception
                   - Assortment or Pack Architecture gaps vs branded competitors
                   - Innovation opportunities that neither branded nor private label are addressing
                   
                5. "ingredient_audit": A list of objects for a table (focus on BRANDED competitors for this):
                   - "ingredient_type": (e.g., "Sweetener", "Preservative", "Protein Source").
                   - "my_brand": What I use.
                   - "branded_competitor_1": "Name: What they use".
                   - "branded_competitor_2": "Name: What they use".
                   - "private_label_baseline": "What private label typically uses".
                   - "implication": "Why this matters for differentiation".
                   
                RETURN ONLY JSON. NO MARKDOWN.
                """

                try:
                    response = model.generate_content(prompt)
                    txt = response.text.strip()
                    txt = re.sub(r'```json\s?|```', '', txt, flags=re.IGNORECASE).replace(',}', '}').replace(',]', ']')
                    
                    # Robust JSON parsing
                    try: 
                        result = json.loads(txt)
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
                        st.write(occ_df.to_html(index=False, escape=False), unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🏷️ Claims Strategy")
                        claims = result.get("claims_strategy", {})
                        st.success(f"**Branded Competitors:** {claims.get('branded_competitor_wins', 'N/A')}")
                        if claims.get('private_label_approach'):
                            st.warning(f"**Private Label Strategy:** {claims.get('private_label_approach', 'N/A')}")
                        st.error(f"**Our Critical Gaps:** {claims.get('my_gaps', 'N/A')}")
                        
                    with c2:
                        st.subheader("🧐 Strategic Questions")
                        questions = result.get("strategic_questions", [])
                        for q in questions:
                            st.warning(f"👉 {q}")

                    st.divider()
                    st.subheader("🔬 Technical Ingredient Audit")
                    ing_audit = result.get("ingredient_audit", [])
                    if ing_audit:
                        ing_df = pd.DataFrame(ing_audit)
                        st.write(ing_df.to_html(index=False, escape=False), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
                    st.write("Debug - Response text:", txt[:500])

        # --- AUDIT TRAIL ---
        st.divider()
        with st.expander("🔍 AI Data Normalization Audit"):
            st.write("This table shows how messy brand data was consolidated into clean Parent Entities and classified by type.")
            audit_df = m_df[['brands', 'parent_company', 'brand_type']].drop_duplicates()
            st.dataframe(audit_df, use_container_width=True)
        
        # --- DATA REALITY CHECK ---
        with st.expander("🛡️ Data Reality Check & Methodology", expanded=False):
            st.markdown("""
            **About the Data Sources & Limitations:**
            
            1.  **Market Data (OpenFoodFacts):** This is an open-source, user-generated database (like Wikipedia for food). 
                * *Risk:* Data may be incomplete, out of date, or contain duplicate user entries. 
                * *Mitigation:* We run an **AI MDM Engine** to merge brand names into parent companies, but SKU counts should be treated as directional proxies, not absolute Nielsen/IRI numbers.
            
            2.  **Demographics (US Census):** Data is sourced from the ACS 5-Year Estimates.
                * *Risk:* This data is highly accurate but trails real-time shifts by 1-2 years. It reflects the chosen Zip Codes, not specific shoppers in a specific store.
            
            3.  **Trend Context:** Analysis is limited strictly to the PDF files you upload.
                * *Risk:* If no files are uploaded, the AI relies on its general training data (cutoff dates apply).
                
            *This tool is a Hypothesis Generator, not a Validation Engine. Always verify critical insights with POS data.*
            """)

else:
    st.info("👈 Please enter your API keys and click 'Run Market Scan' to begin.")
