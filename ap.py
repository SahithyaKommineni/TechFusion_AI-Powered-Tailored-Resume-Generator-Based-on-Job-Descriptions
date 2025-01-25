from flask import Flask, render_template, request, redirect, url_for
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

# Function to extract skills from the job description (looking for technical skills)
def extract_skills_from_job_description(job_description):
    doc = nlp(job_description)
    skills = set()
    for token in doc:
        if token.pos_ == "NOUN" and len(token.text) > 2:
            skills.add(token.text)
    return skills

# Function to compute ATS score using TF-IDF and cosine similarity
def calculate_ats_score(resume_text, job_description_text):
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)
    
    # Use TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
    
    # Compute Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ats_score = round(similarity[0][0] * 100, 2)  # Convert to percentage
    return ats_score

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('fronted.html')

# Route to handle the form submission and redirect to the results page
@app.route('/process_resume', methods=['POST'])
def handle_resume_submission():
    # Get the uploaded resume and job description
    resume_file = request.files['resume']
    job_description = request.form['job-description']
    
    # Save the uploaded resume temporarily to process
    resume_path = 'temp_resume.pdf'
    resume_file.save(resume_path)
    
    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_path)
    
    # Calculate the ATS score
    ats_score = calculate_ats_score(resume_text, job_description)
    
    # Extract job title (for now, we use the first line as a placeholder)
    job_title = "Python Developer"  # This could be extracted or determined dynamically
    
    # Extract career objective (if any)
    career_objective = "Looking for a challenging role in software development."
    
    # Extract skills (this is simplified for now)
    matched_skills = ["Python", "Django", "Flask"]
    unmatched_skills = ["AWS", "Docker"]
    
    # Redirect to the results page and pass the data via URL
    return redirect(url_for('show_results', ats_score=ats_score, job_title=job_title, career_objective=career_objective, matched_skills=matched_skills, unmatched_skills=unmatched_skills))

# Route to display the results on a separate page
@app.route('/results')
def show_results():
    ats_score = request.args.get('ats_score')
    job_title = request.args.get('job_title')
    career_objective = request.args.get('career_objective')
    matched_skills = request.args.getlist('matched_skills')
    unmatched_skills = request.args.getlist('unmatched_skills')
    
    return render_template('ind2.html', ats_score=ats_score, job_title=job_title, career_objective=career_objective, matched_skills=matched_skills, unmatched_skills=unmatched_skills)

if __name__ == '__main__':
    app.run(debug=True)
