# ATS Resume Scorer 📄

An AI-powered tool that analyzes resumes against job descriptions to improve your chances of getting through Applicant Tracking Systems (ATS).

## 🎯 Key Features

- **Smart ATS Score**: Get a detailed 0-100 score based on resume-job description match
- **Keyword Analysis**: Identifies critical missing keywords from your resume
- **AI Suggestions**: Powered by Google's Gemini 1.5 Pro for intelligent improvements
- **Multi-Format Support**: Works with PDF and DOCX resume formats
- **Line-by-Line Analysis**: Detailed suggestions for each line of your resume

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - Google Gemini 1.5 Pro
  - TF-IDF Vectorization
  - scikit-learn
- **Document Processing**: 
  - pdfminer.six
  - python-docx

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Cloud API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ATS-Resume-Scorer.git
cd ATS-Resume-Scorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file
   - Add your Google API key: `GOOGLE_API_KEY=your_key_here`

4. Run the application:
```bash
streamlit run app.py
```

## 📱 Usage

1. Launch the application
2. Paste the job description
3. Upload your resume (PDF/DOCX)
4. Click "Analyze Resume"
5. Review:
   - Your ATS Score
   - Missing Keywords
   - Suggested Improvements

## 💡 How It Works

1. **Keyword Extraction**: Uses TF-IDF to identify key terms from job descriptions
2. **Score Calculation**: Employs weighted keyword matching
3. **AI Analysis**: Leverages Gemini 1.5 Pro for contextual suggestions
4. **Report Generation**: Provides detailed, actionable feedback

