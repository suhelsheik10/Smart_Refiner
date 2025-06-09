# Caprae Capital AI Lead Refiner 🚀

## Overview

Welcome to the **Caprae Capital AI Lead Refiner**, an intelligent tool designed to transform raw lead data into actionable insights for strategic investment. Built in response to the Caprae Capital AI-Readiness Pre-Screening Challenge, this application directly supports Caprae's vision of identifying high-potential businesses ripe for AI-driven transformation and value creation post-acquisition.

Unlike traditional lead generation tools, this solution goes beyond basic data extraction. It leverages advanced natural language processing to assess a company's **AI readiness** through semantic analysis, enriches lead profiles with critical contact information (LinkedIn, general email) scraped directly from websites, and presents these insights through a clean, intuitive interface.

Our focus for this challenge was a **"Quality First"** approach, enhancing the precision of lead qualification and providing immediate, relevant data points for Caprae Capital's unique M&A as a Service model.

## Features

* **Intelligent AI Readiness Scoring:**
    * **Semantic Analysis:** Utilizes a pre-trained `SentenceTransformer` model (`all-MiniLM-L6-v2`) to deeply understand company descriptions and identify their relevance to various AI concepts, providing a nuanced `AI Readiness Score`.
    * **Keyword Matching:** Incorporates traditional keyword matching for identifying specific industry categories and pain points (e.g., "operational efficiency," "cost reduction").
    * **Categorization:** Assigns leads to `AI Readiness Levels` (High, Medium, Low Potential, No Link) for quick prioritization.
* **Automated Contact Information Enrichment:**
    * **Website Scraping:** Automatically visits provided company URLs to extract public `LinkedIn company profiles` and `general contact email addresses`.
    * **Robust Scraping:** Includes error handling for various website issues (timeouts, connection errors, HTTP errors) and employs polite scraping practices (`time.sleep`).
* **Intuitive User Interface (UI):**
    * Built with Streamlit, featuring a professional and user-friendly dark theme.
    * Clear step-by-step process: Upload, Preview, Analyze & Scrape, Explore Results.
    * Interactive data table with filtering, sorting, and direct links to websites and LinkedIn profiles.
    * Summary metrics and visual charts provide a quick overview of lead distribution.
* **Export Functionality:** Download refined lead data as a CSV for seamless integration into CRM systems or further analysis.
* **Integrated Caprae AI Assistant (Chatbot):** A Gemini-powered chatbot is included for general queries about AI, industries, or business strategy, enhancing user engagement and providing instant insights.

## Installation and Setup

Follow these steps to get the AI Lead Refiner up and running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    (Replace `your-username/your-repo-name` with your actual GitHub path)

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **API Key Configuration (Important for AI Assistant):**
    * This application uses the Google Gemini API for its integrated chatbot. For local demonstration purposes, the API key is **hardcoded directly in `app.py`** to ensure immediate functionality without additional setup steps.
    * **Note:** In a production environment, it is strongly recommended to store API keys securely using environment variables or dedicated secrets management services (e.g., Streamlit Secrets for Streamlit Cloud).
    * The current key is `AIzaSyDxZkvUZHpr7h6IEumG0rCCweviazEt4fA` (This will be the key you keep for the video demo).

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open the application at `http://localhost:8501`.

## Usage

1.  **Upload CSV:** Click "Browse files" under "1. Upload Your Data" and select your lead CSV file.
    * **Required Columns:** Ensure your CSV contains at least `company_name`, `website_text_snippet`, and `website_url`. The `industry` column is optional but enhances analysis.
    * **Tip:** For text fields with commas, ensure they are enclosed in double quotes.
2.  **Review Raw Data:** A preview of your uploaded data will be shown.
3.  **Analyze & Scrape:** Click the "Analyze & Scrape Leads" button. A progress bar will indicate the processing status as the tool refines leads and scrapes website information.
4.  **Explore Results:** Once complete, the "Refined Leads: Insights Unlocked!" section will appear, showing summary statistics, charts, and a detailed, filterable table of your refined leads.
5.  **Download:** Use the "Download Refined Leads as CSV" button to save the enhanced dataset.
6.  **AI Assistant:** Interact with the "Caprae AI Assistant" at the bottom of the page for any general questions.

## Example CSV Structure

Your input CSV should look similar to this:

```csv
company_name,website_url,website_text_snippet,industry
"Innovate AI Solutions","[https://www.innovateaisolutions.com](https://www.innovateaisolutions.com)","We provide cutting-edge machine learning and deep learning platforms for enterprise clients. Our focus is on intelligent automation and data-driven decision making.","Technology"
"LogiFlow Inc.","[https://www.logiflow.net](https://www.logiflow.net)","Optimizing supply chain logistics with advanced analytics and predictive modeling to reduce operational costs.","Logistics"
"HealthTech Pioneers","[https://www.healthtechpioneers.org](https://www.healthtechpioneers.org)","Revolutionizing patient care through AI-powered diagnostics and intelligent medical software.","Healthcare"
"Manual Processes Ltd.","[https://www.manualprocltd.com](https://www.manualprocltd.com)","We offer traditional consulting services to improve business operations through proven methodologies.","Consulting"
