import streamlit as st
import pandas as pd
import re
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time

# NEW IMPORTS for Semantic Scoring
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Caprae Capital AI Lead Refiner 🚀")

# --- Semantic Similarity Model Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_embedding_model():
    """Loads and caches the SentenceTransformer model for text embeddings."""
    model_name = 'all-MiniLM-L6-v2' # A good balance of size and performance
    model = SentenceTransformer(model_name)
    return model

# Load the model once at the beginning of the script
embedding_model = load_embedding_model()

# --- AI Concept Sentences and their Embeddings (for semantic comparison) ---
# These sentences represent various aspects of AI relevance.
# Increased variety for better semantic matching.
ai_concept_sentences = {
    "artificial intelligence solutions": 10,
    "machine learning algorithms development": 10,
    "deep learning research": 10,
    "natural language processing systems": 8,
    "computer vision applications": 8,
    "AI strategy consulting and implementation": 12,
    "data science and advanced analytics": 7,
    "predictive modeling and forecasting": 7,
    "automation software and robotics": 6,
    "digital transformation initiatives using AI": 5,
    "cloud computing infrastructure for AI": 4,
    "SaaS platform with integrated AI features": 5,
    "AI-powered product or service innovation": 9,
    "generative AI for content creation and design": 11,
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
    "AI for cybersecurity": 7,
    "AI for marketing automation": 6,
    "AI for human resources and talent management": 5,
}

# Pre-compute embeddings for AI concept sentences once at startup
ai_concept_embeddings = embedding_model.encode(list(ai_concept_sentences.keys()), convert_to_tensor=True)


# --- Core Refinement Logic ---
def define_keywords():
    """
    Defines the keywords for industry categorization and pain point scoring.
    Semantic scoring handles most of the AI relevance now.
    """
    # These AI keywords are kept primarily for populating 'matched_keywords' if explicitly found
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
    """
    Cleans text for analysis by converting to lowercase and removing non-alphanumeric chars.
    Handles NaN values gracefully.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def refine_lead(lead_row, ai_keywords, industry_keywords, pain_point_keywords):
    """
    Analyzes a single lead's text snippet for AI readiness, industry, and pain points.
    Now uses semantic similarity for AI scoring.
    """
    # This string will be used for both semantic embedding and exact phrase matching (pain points)
    full_text_string = clean_text(lead_row.get('website_text_snippet', '') + " " +
                                  lead_row.get('industry', ''))
    words_in_text = full_text_string.split() # For single-word matching (industry)

    industry_categories = []
    pain_point_score = 0
    matched_keywords = set() # This will now collect ALL matched keywords (exact + semantic)

    # --- NEW: Semantic AI Scoring ---
    ai_score = 0 # Initialize ai_score here
    matched_keywords_semantic = set() # To store concepts that score high semantically

    if full_text_string: # Only process if there's text
        lead_embedding = embedding_model.encode(full_text_string, convert_to_tensor=True)
        # Calculate cosine similarity between lead text and all AI concept embeddings
        cosine_scores = util.cos_sim(lead_embedding, ai_concept_embeddings)[0]

        # Iterate through scores and concept sentences to assign score
        for i, score_tensor in enumerate(cosine_scores):
            concept_sentence = list(ai_concept_sentences.keys())[i]
            base_weight = list(ai_concept_sentences.values())[i]
            score_value = score_tensor.item() # Extract scalar value

            similarity_threshold = 0.30 # Tuned based on recent tests; adjust as needed
            # For real-world data, you might fine-tune this after more testing.

            if score_value >= similarity_threshold:
                # Add score based on similarity strength and base weight
                ai_score += base_weight * score_value
                matched_keywords_semantic.add(concept_sentence)

    # --- Existing Industry Keyword Matching ---
    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if keyword in words_in_text: # Check for individual words in the split text
                if industry.capitalize() not in industry_categories:
                    industry_categories.append(industry.capitalize())
                matched_keywords.add(keyword) # Add exact industry keywords

    # --- Existing Pain Point Keyword Matching ---
    for keyword_phrase, score in pain_point_keywords.items():
        if keyword_phrase in full_text_string: # Check for phrases in the full cleaned string
            pain_point_score += score
            matched_keywords.add(keyword_phrase) # Add exact pain point keywords

    # --- Combine all matched keywords (semantic AI + exact industry/pain) ---
    matched_keywords.update(matched_keywords_semantic)

    overall_score = ai_score + pain_point_score

    # Adjust these thresholds based on the new semantic scoring ranges after testing
    if overall_score >= 20: # Example thresholds, tune based on your data's scores
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
    """
    Attempts to scrape LinkedIn URL and a general email from a given website.
    Includes error handling and status reporting.
    """
    linkedin_url = "Not Found"
    general_email = "Not Found"
    status = "Success" # Default status

    if not website_url or not website_url.strip():
        return linkedin_url, general_email, "Invalid/Missing URL"

    # Ensure URL has a scheme (http/https)
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'http://' + website_url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(website_url, headers=headers, timeout=5) # 5 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find LinkedIn link in header/footer or common social icons
        for link in soup.find_all('a', href=True):
            if 'linkedin.com/company/' in link['href'] or 'linkedin.com/in/' in link['href']:
                linkedin_url = link['href']
                break # Found one, good enough for this demo

        # Try to find a general email (very basic check)
        email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text)
        if email_matches:
            # Prioritize info@, contact@, support@ type emails if present, else first one
            for email in email_matches:
                if any(prefix in email for prefix in ['info@', 'contact@', 'support@']):
                    general_email = email
                    break
            if general_email == "Not Found": # If specific not found, take the first one
                general_email = email_matches[0]

    except requests.exceptions.Timeout:
        status = "Timeout Error"
    except requests.exceptions.ConnectionError:
        status = "Connection Error"
    except requests.exceptions.HTTPError as e:
        status = f"HTTP Error {e.response.status_code}"
    except Exception as e:
        status = f"Other Error: {type(e).__name__}"

    time.sleep(0.1) # Be polite, add a small delay between requests
    return linkedin_url, general_email, status


# --- Streamlit Application Layout & Logic ---
st.title("Caprae Capital: AI Lead Refiner �")
st.markdown("""
Welcome to the **AI Lead Refiner**! This intelligent tool empowers Caprae Capital to swiftly identify and prioritize high-potential investment opportunities. By analyzing raw lead data, it intelligently assesses a company's **AI readiness** and **enhances lead profiles with scraped contact information**, transforming broad lists into actionable insights.
""")

st.markdown("---")

# Initialize session state variables for data persistence across reruns
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Your Data 📂")
    st.write("Provide a CSV file containing your lead information. Ensure columns like `company_name`, `website_text_snippet`, and critically, `website_url` are present for optimal analysis and **scraping**.")
    st.info("💡 **Tip:** Ensure any text fields with commas are enclosed in double quotes in your CSV to avoid parsing errors.")

with col2:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")

    # This block processes the data only when a new file is uploaded or the button is explicitly pressed
    if uploaded_file is not None and (uploaded_file.name != st.session_state.uploaded_file_name or st.session_state.processed_data is None):
        st.session_state.uploaded_file_name = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a file with data.")
                st.session_state.processed_data = None  # Clear any previous data
            elif 'website_url' not in df.columns:
                st.error("Missing 'website_url' column in your CSV. This column is required for scraping. Please update your CSV.")
                st.session_state.processed_data = None # Clear any previous data
            else:
                st.success("🎉 File uploaded successfully! Ready for refinement.")
                st.subheader("2. Raw Lead Data Preview 👀")
                st.dataframe(df.head(10))

                st.markdown("---")

                st.header("3. Refine Leads & Discover AI Potential ✨")
                st.markdown("Click the button below to analyze your leads for AI readiness **and fetch additional contact information** from their websites. This process leverages intelligent keyword matching and targeted web scraping.")

                if st.button("Analyze & Scrape Leads", key="analyze_button"): # Process data only if button is pressed
                    st.info("Processing leads... This might take a moment as we visit each website. ⏳")
                    ai_kws, industry_kws, pain_kws = define_keywords()
                    results = []
                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)

                    for i, row in df.iterrows():
                        refined_part = refine_lead(row, ai_kws, industry_kws, pain_kws).to_dict()
                        # Retrieve scrape_status along with LinkedIn and email
                        scraped_linkedin, scraped_email, scrape_status = get_contact_info_from_website(row.get('website_url', ''))
                        combined_row = row.to_dict()
                        combined_row.update(refined_part)
                        combined_row['contact_linkedin_url'] = scraped_linkedin
                        combined_row['contact_email_general'] = scraped_email
                        combined_row['scraping_status'] = scrape_status # Add the status to the row
                        results.append(combined_row)

                        my_bar.progress((i + 1) / len(df), text=f"Scraping & refining: {row.get('company_name', 'N/A')}")

                    my_bar.empty()
                    st.success("Analysis and scraping complete! ✅")
                    st.session_state.processed_data = pd.DataFrame(results) # Store processed data in session state
                    # Trigger a rerun to ensure the display section is rendered immediately after processing
                    st.rerun()
        except Exception as e:
            st.error(f"An error occurred while processing your file. Please ensure it's a valid CSV with expected columns including 'website_url'. Error: {e}")
            st.session_state.processed_data = None # Clear any previous data on error
    elif uploaded_file is None and st.session_state.processed_data is None:
        st.info("Please upload a CSV file to get started with the AI Lead Refiner.")


# Display processed data section only if data exists in session state
if st.session_state.processed_data is not None:
    df_refined = st.session_state.processed_data # Retrieve from session state

    st.subheader("Refined Leads: Insights Unlocked! 📊")

    # Display summary statistics using columns
    col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
    col_sum1.metric("Total Leads Processed", len(df_refined))
    col_sum2.metric("High AI Potential Leads", df_refined[df_refined['ai_readiness_level'].str.contains("High AI Potential")].shape[0])
    col_sum3.metric("Avg. AI Readiness Score", f"{df_refined['ai_readiness_score'].mean():.1f}")
    col_sum4.metric("Leads with LinkedIn Found", df_refined[df_refined['contact_linkedin_url'] != "Not Found"].shape[0])
    col_sum5.metric("Successful Scrapes", df_refined[df_refined['scraping_status'] == "Success"].shape[0]) # New metric for successful scrapes

    st.markdown("---")

    st.subheader("Lead Portfolio Overview 📈")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### AI Readiness Levels Distribution")
        # Ensure the order of levels for consistent chart display
        level_order = ["High AI Potential 🌟", "Medium AI Potential 👍", "Low AI Potential (Needs Nurturing) 🌱", "No Clear AI Link ⛔"]
        level_counts = df_refined['ai_readiness_level'].value_counts().reindex(level_order, fill_value=0)
        st.bar_chart(level_counts)

    with col_chart2:
        st.markdown("##### Identified Industry Categories")
        category_counts = df_refined['identified_categories'].value_counts()
        if 'General' in category_counts.index:
            general_count = category_counts['General']
            other_categories = category_counts.drop('General')
            if len(other_categories) > 5: # Show top 5 plus an 'Other' category if many exist
                top_categories = other_categories.nlargest(5)
                rest_sum = other_categories.nsmallest(len(other_categories) - 5).sum()
                if rest_sum > 0:
                    top_categories['Other'] = rest_sum
                final_categories = top_categories.reindex(top_categories.index.tolist() + ['General'], fill_value=general_count)
            else: # If 5 or fewer, just show them all
                final_categories = category_counts.reindex(category_counts.index.tolist(), fill_value=0)
        else: # If no 'General' category, just show all counts
            final_categories = category_counts

        st.bar_chart(final_categories)

    st.markdown("---")

    st.subheader("Filter & Explore Results 🔍")

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

    st.subheader("4. Detailed Refined Lead Data 📋") # Added this subheader
    st.dataframe(df_filtered_sorted, height=500, use_container_width=True) # Display the filtered and sorted DataFrame

    # Add download button for refined data (Correctly placed now)
    @st.cache_data # Cache this function to prevent re-conversion on every rerun
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
    ---
    ### How to Interpret the Results:
    * **AI Readiness Score:** A quantitative measure (higher is better) indicating how strongly a company's description aligns with AI capabilities, specific industry relevance, and identified pain points.
    * **AI Readiness Level:** Provides a quick qualitative assessment from "No Clear AI Link" to "High AI Potential," guiding your strategic focus.
    * **Identified Categories:** Reveals the primary industry or functional area based on detected keywords.
    * **Matched Keywords:** Lists the exact terms found in the lead's text that contributed to its score, offering direct inspiration for personalized outreach.
    * **Contact LinkedIn URL / Contact Email General:** Directly scraped contact points for easier sales outreach and connection.
    * **Scraping Status:** Indicates the success or failure reason for web scraping.
    """)

st.markdown("---")
st.markdown("✨ Developed for the **Caprae Capital AI-Readiness Pre-Screening Challenge**")

# --- Chatbot Integration ---
st.markdown("---")
st.header("Caprae AI Assistant 💬")
st.markdown("Ask the AI Assistant questions about AI, industries, or anything else!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to interact with the LLM (Gemini API) - Synchronous version for direct call
def generate_response_sync(prompt_text_sync, history_sync):
    payload_sync = {"contents": history_sync}
    apiKey_sync = "AIzaSyDxZkvUZHpr7h6IEumG0rCCweviazEt4fA" # Canvas will provide this if empty for gemini-2.0-flash
    apiUrl_sync = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey_sync}"

    try:
        response_sync = requests.post(
            apiUrl_sync,
            headers={'Content-Type': 'application/json'},
            json=payload_sync,
            timeout=30 # Increased timeout to 30 seconds
        )
        response_sync.raise_for_status() # Raise an exception for HTTP errors
        result_sync = response_sync.json()

        if result_sync.get("candidates") and result_sync["candidates"][0].get("content") and \
           result_sync["candidates"][0]["content"].get("parts") and result_sync["candidates"][0]["content"]["parts"][0].get("text"):
            return result_sync["candidates"][0]["content"].get("parts")[0].get("text")
        else:
            # Log full response for debugging if format is unexpected
            print(f"DEBUG: Unexpected API response format: {result_sync}")
            # Check for 'promptFeedback' which indicates safety filters
            if result_sync.get("promptFeedback") and result_sync["promptFeedback"].get("blockReason"):
                block_reason = result_sync["promptFeedback"]["blockReason"]
                return f"Sorry, I couldn't generate a response. The content was blocked due to: {block_reason}. Please try rephrasing."
            return "Sorry, I couldn't generate a response. The API returned an unexpected format."
    except requests.exceptions.RequestException as e:
        # Log specific request exceptions
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"API HTTP Error: Status Code {e.response.status_code}, Response: {e.response.text}")
        elif isinstance(e, requests.exceptions.ConnectionError):
            print(f"API Connection Error: Could not connect to {apiUrl_sync}. Check internet/firewall. Error: {e}")
        elif isinstance(e, requests.exceptions.Timeout):
            print(f"API Timeout Error: Request to {apiUrl_sync} timed out after 30 seconds. Error: {e}")
        else:
            print(f"API Request Exception: {e}")
        return "Sorry, I'm having trouble connecting to the AI. Please try again later."
    except Exception as e:
        # Log any other unexpected errors
        print(f"Unexpected Error during API call: {e}")
        return "An error occurred while processing your request."

# Chat input at the bottom of the container
if prompt := st.chat_input("Ask the AI Assistant..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("AI thinking..."):
            # Prepare chat history for the sync function call
            current_chat_history_for_llm = []
            for msg in st.session_state.messages: # This includes the latest user message
                current_chat_history_for_llm.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

            # Call the synchronous version
            assistant_response = generate_response_sync(prompt, current_chat_history_for_llm)

        st.markdown(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

