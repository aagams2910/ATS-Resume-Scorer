import streamlit as st
import pdfminer.high_level as pdfminer
import docx2txt
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords_from_jd(jd_text, n_keywords=30):
    """Extract important keywords from job description using TF-IDF and context"""
    # Preprocess text
    processed_jd = preprocess_text(jd_text)
    
    # Extract using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Consider both single words and bigrams
        max_features=n_keywords*2  # Get more candidates than needed
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_jd])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Filter out very common terms and single-character terms
        stop_words = set(stopwords.words('english'))
        filtered_keywords = {}
        for keyword, score in zip(feature_names, scores):
            if len(keyword) > 1 and keyword not in stop_words:
                filtered_keywords[keyword] = score
                
        # Sort by importance score
        return {k: v for k, v in sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    except:
        # Fallback to simple word frequency if TF-IDF fails
        words = processed_jd.split()
        word_freq = Counter(words)
        return {k: v for k, v in word_freq.most_common(n_keywords) if k not in stopwords.words('english') and len(k) > 1}

def extract_skills_from_text(text):
    """Extract potential skills from text using common skill patterns"""
    # Common programming languages, frameworks, tools, etc.
    common_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular", "vue", 
        "node.js", "express", "django", "flask", "tensorflow", "pytorch", "keras",
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "data analysis", "sql", "nosql", "mongodb", "mysql", "postgresql", "aws", 
        "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "agile", "scrum",
        "jira", "rest api", "graphql", "full stack", "frontend", "backend"
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
            
    return found_skills

def score_resume(resume_text, jd_text):
    """Enhanced resume scoring with weighted keyword matching and contextual analysis"""
    # Extract keywords from JD
    keywords = extract_keywords_from_jd(jd_text)
    
    # Preprocess resume text
    processed_resume = preprocess_text(resume_text)
    
    # Calculate semantic match score
    total_weight = sum(keywords.values())
    if total_weight == 0:
        return 0, [], {}
    
    score = 0
    matched_keywords = {}
    missing_keywords = []
    
    for keyword, weight in keywords.items():
        if keyword in processed_resume:
            score += weight
            matched_keywords[keyword] = weight
        else:
            missing_keywords.append(keyword)
    
    # Calculate final score (capped at 100)
    match_percentage = (score / total_weight) * 100
    
    # Extract skills from both documents
    jd_skills = set(extract_skills_from_text(jd_text))
    resume_skills = set(extract_skills_from_text(resume_text))
    
    # Add skills match bonus (up to 15%)
    if jd_skills:
        skills_match_rate = len(resume_skills.intersection(jd_skills)) / len(jd_skills)
        skill_bonus = skills_match_rate * 15
    else:
        skill_bonus = 0
    
    final_score = min(100, match_percentage + skill_bonus)
    
    # Return most relevant missing keywords
    sorted_missing = sorted(missing_keywords, key=lambda k: keywords.get(k, 0), reverse=True)
    top_missing = sorted_missing[:10]  # Return top 10 missing keywords
    
    return final_score, top_missing, matched_keywords

def generate_top_improvements(resume_text, jd_text):
    """Generate top 5 actionable improvements for the resume based on the job description"""
    try:
        prompt = f"""
        As an expert ATS resume analyst, review this resume for a job match. 
        
        JOB DESCRIPTION:
        {jd_text}
        
        RESUME:
        {resume_text}
        
        Provide the TOP 5 most critical, specific improvements this candidate should make to their resume to better match this job description.
        
        
        Format as a numbered list with 5 items. Each suggestion should be practical and specific - not generic advice.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        if "429" in str(e):  # Rate limit error
            return "API rate limit reached. Please try again later."
        else:
            return f"Error generating improvement suggestions: {str(e)}"

def analyze_resume_section(section_text, jd_text):
    """Generate improvement for a specific resume section"""
    try:
        if not section_text or len(section_text.split()) < 5:
            return section_text, "Section too short for analysis"
            
        prompt = f"""
        As an ATS resume expert, improve this resume section to better match the job description:
        
        JOB DESCRIPTION:
        {jd_text}
        
        ORIGINAL SECTION:
        {section_text}
        
        Provide ONLY:
        1. An improved version of the section (make meaningful changes, not just small rewording)
        2. A brief explanation of why these changes help match the job description better
        
        Format your response exactly like this:
        IMPROVED_VERSION: [your improved text]
        EXPLANATION: [your explanation]
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Parse the response
        if "IMPROVED_VERSION:" in response_text and "EXPLANATION:" in response_text:
            parts = response_text.split("EXPLANATION:")
            improved_text = parts[0].replace("IMPROVED_VERSION:", "").strip()
            explanation = parts[1].strip()
            return improved_text, explanation
        else:
            return section_text, "Unable to parse suggestion"
            
    except Exception as e:
        if "429" in str(e):  # Rate limit error
            return section_text, "API rate limit reached. Please try again in a minute."
        else:
            return section_text, f"Error generating suggestion: {str(e)}"

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    return pdfminer.extract_text(file_path)

def extract_resume_sections(resume_text):
    """Split resume into logical sections for targeted improvement"""
    # Simple heuristic: split by lines with all caps or lines ending with colon
    lines = resume_text.split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        stripped_line = line.strip()
        # Check if this looks like a section header
        if stripped_line and (stripped_line.isupper() or stripped_line.endswith(':') or len(stripped_line) < 30 and stripped_line[0:1].isupper()):
            # Save previous section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
            # Start new section
            current_section = [stripped_line]
        else:
            current_section.append(stripped_line)
            
    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))
        
    # Filter out very short sections
    return [section for section in sections if len(section.split()) >= 5]

def main():
    st.set_page_config(page_title="ATS Resume Scorer", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { padding: 2rem }
    .stProgress > div > div > div { background-color: #4CAF50 }
    .keyword-tag {
        background-color: #2c3e50; 
        color: white; 
        padding: 0.3rem 0.6rem; 
        margin: 0.2rem; 
        border-radius: 15px;
        display: inline-block;
    }
    .matched-tag {
        background-color: #27ae60;
    }
    .missing-tag {
        background-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 ATS Resume Scorer")
    st.subheader("Analyze how well your resume matches the job description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jd_text = st.text_area("📝 Enter Job Description", height=300, 
                              placeholder="Paste the full job description here...")
        
    with col2:
        uploaded_file = st.file_uploader("📄 Upload Your Resume", type=["pdf", "docx"],
                                        help="Upload your resume in PDF or DOCX format")
        
    if st.button("🔍 Analyze Resume") and jd_text and uploaded_file:
        try:
            with st.spinner("Analyzing your resume against the job description..."):
                # Extract resume text
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = docx2txt.process(uploaded_file)
                
                # Calculate match score
                ats_score, missing_keywords, matched_keywords = score_resume(resume_text, jd_text)
                
                # Generate overall improvement suggestions
                improvement_suggestions = generate_top_improvements(resume_text, jd_text)
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Score visualization
                score_col1, score_col2 = st.columns([1, 3])
                with score_col1:
                    st.markdown(f"""
                    <div style="text-align: center; 
                                background-color: {'#27ae60' if ats_score >= 70 else '#e67e22' if ats_score >= 50 else '#e74c3c'}; 
                                color: white; 
                                padding: 1rem; 
                                border-radius: 10px; 
                                font-size: 2rem;
                                font-weight: bold;">
                        {ats_score:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                with score_col2:
                    st.progress(ats_score/100)
                    if ats_score >= 70:
                        st.success("Your resume is well-matched to this job description.")
                    elif ats_score >= 50:
                        st.warning("Your resume has moderate alignment with this job description. Consider the suggestions below.")
                    else:
                        st.error("Your resume needs significant improvement to match this job description.")
                
                # Keywords section
                st.markdown("## 🔑 Keyword Analysis")
                keyword_col1, keyword_col2 = st.columns(2)
                
                with keyword_col1:
                    st.markdown("### ✅ Matched Keywords")
                    if matched_keywords:
                        # Sort matched keywords by importance
                        sorted_matched = sorted(matched_keywords.items(), key=lambda x: x[1], reverse=True)
                        matched_html = " ".join([f"<span class='keyword-tag matched-tag'>{kw}</span>" 
                                              for kw, _ in sorted_matched])
                        st.markdown(f"<div style='line-height: 3;'>{matched_html}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("No significant keywords matched.")
                
                with keyword_col2:
                    st.markdown("### ❌ Missing Keywords")
                    if missing_keywords:
                        missing_html = " ".join([f"<span class='keyword-tag missing-tag'>{kw}</span>" 
                                              for kw in missing_keywords])
                        st.markdown(f"<div style='line-height: 3;'>{missing_html}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("No significant keywords missing.")
                
                # Top 5 improvement suggestions
                st.markdown("## 🚀 Top 5 Improvement Suggestions")
                st.markdown(improvement_suggestions)
                
                # Section-by-section analysis
                st.markdown("## 📝 Detailed Section Analysis")
                st.markdown("Expand each section to see specific improvement suggestions.")
                
                sections = extract_resume_sections(resume_text)
                for i, section in enumerate(sections):
                    if len(section.split()) >= 10:  # Only analyze substantial sections
                        improved_section, explanation = analyze_resume_section(section, jd_text)
                        
                        # Determine a good title for the expander
                        first_line = section.strip().split('\n')[0]
                        title = first_line if len(first_line) < 50 else first_line[:47] + "..."
                        
                        with st.expander(f"Section {i+1}: {title}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original:**")
                                st.text(section)
                            with col2:
                                st.markdown("**Improved Version:**")
                                st.text(improved_section)
                            st.markdown("**Why make these changes?**")
                            st.markdown(explanation)
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            if "429" in str(e):
                st.warning("API rate limit reached. Please wait a minute and try again.")

if __name__ == "__main__":
    main()
