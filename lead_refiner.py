import csv
import re
import pandas as pd # Needed for pd.Series and pd.isna
import time # For any potential future scraping, though not used in this refine-only script
import requests # For web scraping functionality
from bs4 import BeautifulSoup # For parsing HTML in web scraping
from urllib.parse import urlparse, urljoin # For URL handling in web scraping

# NEW IMPORTS for Semantic Scoring
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Semantic Similarity Model Loading ---
# For a standalone script, caching is less critical but good practice if run multiple times
def load_embedding_model():
    """Loads the SentenceTransformer model for text embeddings."""
    model_name = 'all-MiniLM-L6-v2' # Consistent model name
    model = SentenceTransformer(model_name)
    return model

# Load the model once when the script is imported/run
embedding_model = load_embedding_model()

# --- AI Concept Sentences and their Embeddings (for semantic comparison) ---
# These should be consistent with app.py's definition if app.py imports this
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

# --- Core Refinement Logic (Same as before) ---
def define_keywords():
    """Defines and returns dictionaries of keywords for AI, industries, and pain points."""
    # These are your exact match keywords, primarily for industry and pain points
    # For AI, semantic scoring is now primary, but these can still contribute
    ai_keywords = { # These will primarily be used for "matched_keywords" if explicitly found
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
    """Cleans input text by converting to lowercase and removing special characters."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def refine_lead(lead_row, ai_keywords, industry_keywords, pain_point_keywords):
    """
    Refines a single lead row by calculating AI readiness, identifying industries,
    and matching keywords.
    """
    full_text_string = clean_text(lead_row.get('website_text_snippet', '') + " " +
                                  lead_row.get('industry', ''))
    words_in_text = full_text_string.split()
    industry_categories = []
    pain_point_score = 0
    matched_keywords = set()
    ai_score = 0
    matched_keywords_semantic = set()

    # Semantic similarity scoring for AI concepts
    if full_text_string:
        lead_embedding = embedding_model.encode(full_text_string, convert_to_tensor=True)
        cosine_scores = util.cos_sim(lead_embedding, ai_concept_embeddings)[0]
        for i, score_tensor in enumerate(cosine_scores):
            concept_sentence = list(ai_concept_sentences.keys())[i]
            base_weight = list(ai_concept_sentences.values())[i]
            score_value = score_tensor.item()
            similarity_threshold = 0.30 # Threshold for considering a semantic match
            if score_value >= similarity_threshold:
                ai_score += base_weight * score_value
                matched_keywords_semantic.add(concept_sentence)

    # Exact match scoring for industry and pain points
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

    # Combine all matched keywords
    matched_keywords.update(matched_keywords_semantic)
    overall_score = ai_score + pain_point_score

    # Determine AI readiness level based on overall score
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
    """
    Attempts to scrape LinkedIn and general email from a given website URL.
    Returns default values if info is not found or an error occurs.
    """
    linkedin_url = "Not Found"
    general_email = "Not Found"
    status = "Success" # Default status

    if not website_url or not website_url.strip():
        return linkedin_url, general_email, "Invalid/Missing URL"

    # Prepend 'http://' if scheme is missing for requests to work
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'http://' + website_url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(website_url, headers=headers, timeout=5) # 5-second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find LinkedIn URL
        for link in soup.find_all('a', href=True):
            if 'linkedin.com/company/' in link['href'] or 'linkedin.com/in/' in link['href']:
                linkedin_url = link['href']
                break # Found one, no need to continue searching

        # Find general email addresses (info@, contact@, support@ preferred)
        email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text)
        if email_matches:
            # Prioritize common contact emails
            for email in email_matches:
                if any(prefix in email for prefix in ['info@', 'contact@', 'support@']):
                    general_email = email
                    break
            # If no common contact email found, just take the first one
            if general_email == "Not Found":
                general_email = email_matches[0]

    except requests.exceptions.Timeout:
        status = "Timeout Error"
    except requests.exceptions.ConnectionError:
        status = "Connection Error"
    except requests.exceptions.HTTPError as e:
        status = f"HTTP Error {e.response.status_code}"
    except Exception as e:
        # Catch any other unexpected errors during scraping
        status = f"Other Error: {type(e).__name__}"
    
    time.sleep(0.1) # Be polite and avoid overwhelming servers
    return linkedin_url, general_email, status

# If you were to run this script directly, you might add a main block like this:
if __name__ == "__main__":
    print("This script provides functions for lead refinement and web scraping.")
    print("It is typically imported by another script (like app.py) for use.")
    # Example usage (for testing this script independently):
    # test_data = pd.DataFrame([
    #     {'company_name': 'AI Innovators Inc.', 'website_text_snippet': 'We build cutting-edge artificial intelligence solutions for businesses.', 'industry': 'Technology', 'website_url': 'http://www.aiinnovators.com'},
    #     {'company_name': 'Global Logistics Co.', 'website_text_snippet': 'Optimizing supply chains with advanced analytics.', 'industry': 'Logistics', 'website_url': 'http://www.globallogistics.com'}
    # ])
    #
    # ai_kws, industry_kws, pain_kws = define_keywords()
    #
    # refined_leads_list = []
    # for index, row in test_data.iterrows():
    #     refined_row = refine_lead(row, ai_kws, industry_kws, pain_kws).to_dict()
    #     linkedin, email, scrape_status = get_contact_info_from_website(row['website_url'])
    #     full_row = row.to_dict()
    #     full_row.update(refined_row)
    #     full_row['contact_linkedin_url'] = linkedin
    #     full_row['contact_email_general'] = email
    #     full_row['scraping_status'] = scrape_status
    #     refined_leads_list.append(full_row)
    #
    # refined_df = pd.DataFrame(refined_leads_list)
    # print("\nRefined Leads DataFrame:")
    # print(refined_df)

