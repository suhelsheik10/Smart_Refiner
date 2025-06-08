import streamlit as st
import pandas as pd
import re
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
import os
# from dotenv import load_dotenv # Commented out: .env file loading is not working for this local setup

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="Caprae Capital AI Lead Refiner 🚀",
    page_icon="📈"
)

# --- CUSTOM CSS FOR APP AESTHETICS (Subtle Enhancements for Dark Theme) ---
st.markdown(
    """
    <style>
    /* Main app container background with a subtle gradient matching dark theme */
    .stApp {
        background: linear-gradient(to bottom right, #121212, #212121); /* Dark subtle gradient */
    }

    /* Primary button styling - consistent with config.toml dark theme */
    div.stButton > button:first-child {
        background-color: #BB86FC; /* Primary accent color (soft purple) */
        color: #121212; /* Dark text on bright button */
        border-radius: 10px;
        border: none;
        padding: 12px 25px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* More pronounced shadow on dark */
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }

    div.stButton > button:first-child:hover {
        background-color: #9A50F9; /* Darker purple on hover */
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }

    /* Custom styling for st.info alerts (adjusting for dark theme) */
    div.stAlert.stAlert--info {
        background-color: #263238; /* Darker blue-grey background */
        border-left: 6px solid #81D4FA; /* Lighter blue border for contrast */
        color: #BBDEFB; /* Lighter text color */
        border-radius: 8px;
        padding: 12px 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Custom styling for st.success alerts (adjusting for dark theme) */
    div.stAlert.stAlert--success {
        background-color: #2E7D32; /* Darker green background */
        border-left: 6px solid #A5D6A7; /* Lighter green border for contrast */
        color: #C8E6C9; /* Lighter text color */
        border-radius: 8px;
        padding: 12px 15px;
        box_shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Adjust margins for Streamlit components to ensure better spacing */
    .stHorizontalBlock {
        margin-bottom: 1rem;
    }
    .stSelectbox, .stFileUploader, .stCheckbox {
        margin-bottom: 0.8rem;
    }
    .stMetric {
        border: 1px solid #333333; /* Darker border for metrics */
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 1rem;
        background-color: #212121; /* Slightly lighter dark for metric cards */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #E0E0E0; /* Light text for metrics */
    }
    .stMetric > div > div:first-child { /* Metric label */
        color: #BB86FC; /* Primary color for labels */
        font-weight: bold;
    }
    .stMetric > div > div:nth-child(2) { /* Metric value */
        color: #E0E0E0; /* Light grey for values */
    }

    /* Custom styling for st.container with border=True */
    .stContainer {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4); /* More prominent shadow on dark */
        background-color: #1E1E1E; /* Secondary background color for content cards */
        color: #E0E0E0; /* Light text for containers */
    }
    .stContainer h1, .stContainer h2, .stContainer h3, .stContainer h4, .stContainer h5, .stContainer h6 {
        color: #BB86FC; /* Use primary color for headers within containers */
    }
    .stContainer p, .stContainer li {
        color: #B0B0B0; /* Slightly dimmer light grey for body text in containers */
    }
    
    /* Better styling for the markdown intro text */
    p {
        line-height: 1.6;
        font-size: 1.05rem;
        color: #E0E0E0; /* Light text for global paragraphs */
    }

    /* Adjusting markdown headers for better visual hierarchy on dark theme */
    h1 { font-size: 2.5rem; color: #BB86FC; margin-bottom: 1.5rem; } /* Main title */
    h2 { font-size: 2rem; color: #E0E0E0; margin-top: 2rem; margin-bottom: 1rem; }
    h3 { font-size: 1.75rem; color: #444444; margin-top: 1.5rem; margin-bottom: 0.8rem; }
    h4 { font-size: 1.5rem; color: #555555; margin-top: 1.2rem; margin-bottom: 0.6rem; }
    h5 { font-size: 1.25rem; color: #BB86FC; margin-top: 1rem; margin-bottom: 0.5rem; }

    /* Centering the chatbot input */
    .stChatInput {
        text-align: center;
    }
    .stChatInput > div > div {
        margin: 0 auto;
        max-width: 700px;
    }
    /* Chat messages text color */
    .stChatMessage {
        color: #E0E0E0; /* Light grey for chat messages */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Semantic Similarity Model Loading ---
from sentence_transformers import SentenceTransformer, util
import numpy as np

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model for text embeddings."""
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model

embedding_model = load_embedding_model()

# --- AI Concept Sentences and their Embeddings (for semantic comparison) ---
ai_concept_sentences = {
    "artificial intelligence solutions": 10, "machine learning algorithms development": 10,
    "deep learning research": 10, "natural language processing systems": 8,
    "computer vision applications": 8, "AI strategy consulting and implementation": 12,
    "data science and advanced analytics": 7, "predictive modeling and forecasting": 7,
    "automation software and robotics": 6, "digital transformation initiatives using AI": 5,
    "cloud computing infrastructure for AI": 4, "SaaS platform with integrated AI features": 5,
    "AI-powered product or service innovation": 9, "generative AI for content creation and design": 11,
    "large language model (LLM) deployment and fine-tuning": 9,
    "intelligent automation for business processes": 8,
    "data driven decision making with AI tools": 6,
    "neural networks and deep learning architectures": 9,
    "computer vision for image and video analysis": 8,
    "speech recognition and natural language understanding": 7,
    "AI for fraud detection and prevention": 6,
    "AI-enhanced customer experience and support": 7,
    "personalized recommendations and content delivery": 7,
    "supply chain optimization with AI": 6,
    "smart manufacturing and industry 4.0 solutions": 6,
    "robotics and AI integration": 8,
    "AI for operational efficiency improvements": 7,
    "AI for business growth and innovation": 7,
    "AI applications in healthcare and biotech": 9,
    "AI in finance and fintech solutions": 9,
    "AI for retail analytics and e-commerce": 8,
    "AI in renewable energy and smart grids": 8,
    "ethical AI and responsible AI development": 5,
    "AI talent development and training": 4,
    "AI for cybersecurity": 7, "AI for marketing automation": 6,
    "AI for human resources and talent management": 5,
}
ai_concept_embeddings = embedding_model.encode(list(ai_concept_sentences.keys()), convert_to_tensor=True)

# --- Core Refinement Logic ---
def define_keywords():
    ai_keywords = {
        "artificial intelligence": 10, "machine learning": 10, "deep learning": 10, "NLP": 8,
        "computer vision": 8, "AI strategy": 12, "data science": 7, "predictive analytics": 7,
        "automation software": 6, "digital transformation": 5, "cloud computing": 4,
        "SaaS solution": 5, "AI-powered": 9, "generative AI": 11, "large language model": 9,
    }
    industry_keywords = {
        "technology": {"tech", "software", "IT", "platform", "SaaS", "cloud"},
        "healthcare": {"healthtech", "medical", "patient management", "diagnostics", "hospital", "pharma"},
        "finance": {"fintech", "financial services", "investment", "fraud detection", "banking", "trading"},
        "logistics": {"supply chain", "warehousing", "logistics management", "freight", "shipping"},
        "manufacturing": {"production line", "industrial automation", "factory", "assembly"},
        "retail": {"e-commerce", "customer experience", "online store", "merchandising"},
        "energy": {"renewable energy", "green technology", "energy efficiency", "solar", "wind"},
    }
    pain_point_keywords = {
        "operational efficiency": 6, "cost reduction": 5, "streamline operations": 6,
        "inefficient workflows": 7, "manual processes": 8, "decision-making": 4,
        "growth opportunities": 3, "customer churn": 7, "scalability issues": 6,
        "data entry": 7, "resource optimization": 5, "market analysis": 4,
    }
    return ai_keywords, industry_keywords, pain_point_keywords

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def refine_lead(lead_row, ai_keywords, industry_keywords, pain_point_keywords):
    full_text_string = clean_text(lead_row.get('website_text_snippet', '') + " " +
                                  lead_row.get('industry', ''))
    words_in_text = full_text_string.split()
    industry_categories = []
    pain_point_score = 0
    matched_keywords = set()
    ai_score = 0
    matched_keywords_semantic = set()

    if full_text_string:
        lead_embedding = embedding_model.encode(full_text_string, convert_to_tensor=True)
        cosine_scores = util.cos_sim(lead_embedding, ai_concept_embeddings)[0]
        for i, score_tensor in enumerate(cosine_scores):
            concept_sentence = list(ai_concept_sentences.keys())[i]
            base_weight = list(ai_concept_sentences.values())[i]
            score_value = score_tensor.item()
            similarity_threshold = 0.30
            if score_value >= similarity_threshold:
                ai_score += base_weight * score_value
                matched_keywords_semantic.add(concept_sentence)

    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if keyword in words_in_text:
                if industry.capitalize() not in industry_categories:
                    industry_categories.append(industry.capitalize())
                matched_keywords.add(keyword)

    for keyword_phrase, score in pain_point_keywords.items():
        if keyword_phrase in full_text_string:
            pain_point_score += score
            matched_keywords.add(keyword_phrase)

    matched_keywords.update(matched_keywords_semantic)
    overall_score = ai_score + pain_point_score

    if overall_score >= 20:
        ai_readiness_level = "High AI Potential 🌟"
    elif overall_score >= 10:
        ai_readiness_level = "Medium AI Potential 👍"
    elif overall_score >= 3:
        ai_readiness_level = "Low AI Potential (Needs Nurturing) 🌱"
    else:
        ai_readiness_level = "No Clear AI Link ⛔"

    return pd.Series({
        'ai_readiness_score': overall_score,
        'ai_readiness_level': ai_readiness_level,
        'identified_categories': ", ".join(industry_categories) if industry_categories else "General",
        'matched_keywords': ", ".join(sorted(list(matched_keywords))) if matched_keywords else "None"
    })

def get_contact_info_from_website(website_url):
    linkedin_url = "Not Found"
    general_email = "Not Found"
    status = "Success"

    if not website_url or not website_url.strip():
        return linkedin_url, general_email, "Invalid/Missing URL"
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'http://' + website_url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(website_url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if 'linkedin.com/company/' in link['href'] or 'linkedin.com/in/' in link['href']:
                linkedin_url = link['href']
                break
        email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text)
        if email_matches:
            for email in email_matches:
                if any(prefix in email for prefix in ['info@', 'contact@', 'support@']):
                    general_email = email
                    break
            if general_email == "Not Found":
                general_email = email_matches[0]

    except requests.exceptions.Timeout:
        status = "Timeout Error"
    except requests.exceptions.ConnectionError:
        status = "Connection Error"
    except requests.exceptions.HTTPError as e:
        status = f"HTTP Error {e.response.status_code}"
    except Exception as e:
        status = f"Other Error: {type(e).__name__}"
    time.sleep(0.1)
    return linkedin_url, general_email, status

# --- Streamlit Application Layout & Logic ---
st.image("caprae_capital_logo.png", width=150) # CORRECTED LOGO FILENAME HERE
st.title("Caprae Capital: AI Lead Refiner 🚀")
st.markdown("""
<p style='font-size:1.1rem; line-height:1.6; color:#E0E0E0;'>
Welcome to the <b>AI Lead Refiner</b>! This intelligent tool empowers Caprae Capital to swiftly identify and prioritize high-potential investment opportunities. By analyzing raw lead data, it intelligently assesses a company's <b>AI readiness</b> and <b>enhances lead profiles with scraped contact information</b>, transforming broad lists into actionable insights.
</p>
""", unsafe_allow_html=True)

st.divider()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# Main content area columns
col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True): # Card-like container for upload
        st.header("1. Upload Your Data 📂")
        st.write("Provide a CSV file containing your lead information. Ensure columns like `company_name`, `website_text_snippet`, and critically, `website_url` are present for optimal analysis and **scraping**.")
        st.info("💡 **Tip:** Ensure any text fields with commas are enclosed in double quotes in your CSV to avoid parsing errors.")
with col2:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_file is not None and (uploaded_file.name != st.session_state.uploaded_file_name or st.session_state.processed_data is None):
        st.session_state.uploaded_file_name = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a file with data.")
                st.session_state.processed_data = None
            elif 'website_url' not in df.columns:
                st.error("Missing 'website_url' column in your CSV. This column is required for scraping. Please update your CSV.")
                st.session_state.processed_data = None
            else:
                st.success("🎉 File uploaded successfully! Ready for refinement.")
                st.subheader("2. Raw Lead Data Preview 👀")
                st.dataframe(df.head(10), use_container_width=True) # Ensure preview also uses full width
                st.divider()
                with st.container(border=True): # Card-like container for refine button
                    st.header("3. Refine Leads & Discover AI Potential ✨")
                    st.markdown("Click the button below to analyze your leads for AI readiness **and fetch additional contact information** from their websites. This process leverages intelligent keyword matching and targeted web scraping.")
                    if st.button("Analyze & Scrape Leads", key="analyze_button", use_container_width=True): # Button fills container
                        st.info("Processing leads... This might take a moment as we visit each website. ⏳")
                        ai_kws, industry_kws, pain_kws = define_keywords()
                        results = []
                        progress_text = "Operation in progress. Please wait."
                        my_bar = st.progress(0, text=progress_text)
                        for i, row in df.iterrows():
                            refined_part = refine_lead(row, ai_kws, industry_kws, pain_kws).to_dict()
                            scraped_linkedin, scraped_email, scrape_status = get_contact_info_from_website(row.get('website_url', ''))
                            combined_row = row.to_dict()
                            combined_row.update(refined_part)
                            combined_row['contact_linkedin_url'] = scraped_linkedin
                            combined_row['contact_email_general'] = scraped_email
                            combined_row['scraping_status'] = scrape_status
                            results.append(combined_row)
                            my_bar.progress((i + 1) / len(df), text=f"Scraping & refining: {row.get('company_name', 'N/A')}")
                        my_bar.empty()
                        st.success("Analysis and scraping complete! ✅")
                        st.session_state.processed_data = pd.DataFrame(results)
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred while processing your file. Please ensure it's a valid CSV with expected columns including 'website_url'. Error: {e}")
            st.session_state.processed_data = None
    elif uploaded_file is None and st.session_state.processed_data is None:
        st.info("Please upload a CSV file to get started with the AI Lead Refiner.")

# Only show results if data has been processed
if st.session_state.processed_data is not None:
    df_refined = st.session_state.processed_data
    st.markdown("## Refined Leads: Insights Unlocked! 📊") # Changed to H2
    
    # Display summary statistics using columns with enhanced st.metric styling
    col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
    col_sum1.metric("Total Leads", len(df_refined), help="Total number of leads processed")
    col_sum2.metric("High AI Potential", df_refined[df_refined['ai_readiness_level'].str.contains("High AI Potential")].shape[0], help="Number of leads with High AI Potential")
    col_sum3.metric("Avg. AI Score", f"{df_refined['ai_readiness_score'].mean():.1f}", help="Average AI Readiness Score across all leads")
    col_sum4.metric("LinkedIn Found", df_refined[df_refined['contact_linkedin_url'] != "Not Found"].shape[0], help="Number of leads where a LinkedIn URL was found")
    col_sum5.metric("Successful Scrapes", df_refined[df_refined['scraping_status'] == "Success"].shape[0], help="Number of websites successfully scraped")

    st.divider()

    st.markdown("### Lead Portfolio Overview 📈") # Changed to H3 for better hierarchy

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### AI Readiness Levels Distribution") # Changed to H5
        level_order = ["High AI Potential 🌟", "Medium AI Potential 👍", "Low AI Potential (Needs Nurturing) 🌱", "No Clear AI Link ⛔"]
        level_counts = df_refined['ai_readiness_level'].value_counts().reindex(level_order, fill_value=0)
        st.bar_chart(level_counts)
    with col_chart2:
        st.markdown("##### Identified Industry Categories") # Changed to H5
        category_counts = df_refined['identified_categories'].value_counts()
        if 'General' in category_counts.index:
            general_count = category_counts['General']
            other_categories = category_counts.drop('General')
            if len(other_categories) > 5:
                top_categories = other_categories.nlargest(5)
                rest_sum = other_categories.nsmallest(len(other_categories) - 5).sum()
                if rest_sum > 0:
                    top_categories['Other'] = rest_sum
                final_categories = top_categories.reindex(top_categories.index.tolist() + ['General'], fill_value=general_count)
            else:
                final_categories = category_counts.reindex(category_counts.index.tolist(), fill_value=0)
        else:
            final_categories = category_counts
        st.bar_chart(final_categories)
    st.divider()

    with st.expander("Filter & Explore Detailed Results 🔍"): # Renamed expander title
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            ai_levels = ["All Levels"] + sorted(df_refined['ai_readiness_level'].unique().tolist())
            selected_level = st.selectbox("Filter by AI Readiness Level:", ai_levels, key="level_filter")
        with col_filter2:
            show_only_with_linkedin = st.checkbox("Show only leads with LinkedIn URL", key="linkedin_filter")
        df_filtered = df_refined.copy()
        if selected_level != "All Levels":
            df_filtered = df_filtered[df_filtered['ai_readiness_level'] == selected_level]
        if show_only_with_linkedin:
            df_filtered = df_filtered[df_filtered['contact_linkedin_url'] != "Not Found"]
        df_filtered_sorted = df_filtered.sort_values(by='ai_readiness_score', ascending=False)
        st.subheader("Detailed Refined Lead Data 📋") # Changed to H3
        st.dataframe(
            df_filtered_sorted,
            height=500,
            use_container_width=True,
            column_config={
                "website_url": st.column_config.LinkColumn(
                    "Website URL", display_text="Visit", help="Company Website"
                ),
                "contact_linkedin_url": st.column_config.LinkColumn(
                    "LinkedIn", display_text="Connect", help="LinkedIn Profile/Company Page"
                ),
                "ai_readiness_score": st.column_config.NumberColumn(
                    "AI Score", format="%.2f", help="AI Readiness Score (higher is better)"
                ),
                "scraping_status": st.column_config.Column(
                    "Scraping Status", help="Status of website scraping attempt"
                ),
                # Add more relevant column_config for other columns like 'company_name', etc.
            }
        )
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')
        csv_download = convert_df_to_csv(df_filtered_sorted)
        st.download_button(
            label="Download Refined Leads as CSV ⬇️",
            data=csv_download,
            file_name="caprae_refined_leads.csv",
            mime="text/csv",
            key="download_button"
        )
    
    st.markdown("""
    <div style='background-color:#1E1E1E; border-radius:10px; padding:20px; margin-top:2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.3);'>
        <h4 style='color:#BB86FC; margin-top:0.5rem; margin-bottom:1rem;'>How to Interpret the Results:</h4>
        <ul style='list-style-type: disc; padding-left:25px; line-height:1.8; color:#B0B0B0;'>
            <li><b>AI Readiness Score:</b> A quantitative measure (higher is better) indicating how strongly a company's description aligns with AI capabilities, specific industry relevance and identified pain points.</li>
            <li><b>AI Readiness Level:</b> Provides a quick qualitative assessment from "No Clear AI Link" to "High AI Potential," guiding your strategic focus.</li>
            <li><b>Identified Categories:</b> Reveals the primary industry or functional area based on detected keywords.</li>
            <li><b>Matched Keywords:</b> Lists the exact terms found in the lead's text that contributed to its score, offering direct inspiration for personalized outreach.</li>
            <li><b>Contact LinkedIn URL / Contact Email General:</b> Directly scraped contact points for easier sales outreach and connection.</li>
            <li><b>Scraping Status:</b> Indicates the success or failure reason for website scraping.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


st.divider()
st.markdown("<p style='text-align:center; color:#B0B0B0; font-size:0.9rem;'>✨ Developed for the <b>Caprae Capital AI-Readiness Pre-Screening Challenge</b></p>", unsafe_allow_html=True)
st.divider()

# Chatbot Integration
st.header("Caprae AI Assistant 💬")
st.markdown("<p style='font-size:1rem; color:#B0B0B0;'>Ask the AI Assistant questions about AI, industries, or anything else!</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_response_sync(prompt_text_sync, history_sync):
    # Retrieve API key: HARDCODED FOR LOCAL FUNCTIONALITY (INSECURE for deployment)
    api_key = "AIzaSyDxZkvUZHpr7h6IEumG0rCCweviazEt4fA" # YOUR API KEY HARDCODED HERE
    
    if not api_key: 
        print("Error: GOOGLE_API_KEY is not set (hardcoded key is empty).")
        return "API key not found. Please ensure the API key is correctly hardcoded for local testing."

    payload_sync = {"contents": history_sync}
    apiUrl_sync = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        response_sync = requests.post(
            apiUrl_sync,
            headers={'Content-Type': 'application/json'},
            json=payload_sync,
            timeout=30
        )
        response_sync.raise_for_status()
        result_sync = response_sync.json()

        # DEBUG LINE: Print the full API response for debugging
        print(f"DEBUG: Full API Response: {result_sync}")

        if result_sync.get("candidates") and result_sync["candidates"][0].get("content") and \
           result_sync["candidates"][0]["content"].get("parts") and result_sync["candidates"][0]["content"]["parts"][0].get("text"):
            # CORRECTED LINE: Use square bracket notation for dictionary access
            return result_sync["candidates"][0]["content"]["parts"][0]["text"]
        else:
            if result_sync.get("promptFeedback") and result_sync["promptFeedback"].get("blockReason"):
                block_reason = result_sync["promptFeedback"]["blockReason"]
                return f"Sorry, I couldn't generate a response. The content was blocked due to: {block_reason}. Please try rephrasing."
            elif result_sync.get("error"):
                error_details = result_sync["error"].get("message", "Unknown API error.")
                return f"Sorry, the AI encountered an error: {error_details}"
            return "Sorry, I couldn't generate a response. The API returned an unexpected format."
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"API HTTP Error: Status Code {e.response.status_code}, Response: {e.response.text}")
            return f"Sorry, the AI encountered an HTTP error: Status {e.response.status_code}. Please check your API key or ensure billing is enabled in Google Cloud."
        elif isinstance(e, requests.exceptions.ConnectionError):
            print(f"API Connection Error: Could not connect to {apiUrl_sync}. Check internet/firewall. Error: {e}")
            return "Sorry, I'm having trouble connecting to the AI. Please check your internet connection."
        elif isinstance(e, requests.exceptions.Timeout):
            print(f"API Timeout Error: Request to {apiUrl_sync} timed out after 30 seconds. Error: {e}")
            return "Sorry, the AI response timed out. Please try again."
        else:
            print(f"API Request Exception: {e}")
            return f"Sorry, an unexpected network error occurred: {e}"
    except Exception as e:
        print(f"Unexpected Error during API call: {e}")
        return "An internal error occurred while processing your request."

if prompt := st.chat_input("Ask the AI Assistant..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI thinking..."):
            current_chat_history_for_llm = []
            for msg in st.session_state.messages:
                current_chat_history_for_llm.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
            assistant_response = generate_response_sync(prompt, current_chat_history_for_llm)
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
