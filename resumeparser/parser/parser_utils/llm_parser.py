import requests
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def extract_resume_data(resume_text):
    prompt = f"""
You are an AI resume parser. Extract the following fields from this resume text and return a valid JSON.
Calculate total professional experience in years by analyzing the work experience timeline. Use the current date for "Present" if needed.
Return output in exactly this format:

{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "experience_years":,
  "skills": [],
  "current_role": "",
  "company": "",
  "education": [],
  "projects": [
    {{
      "title": "",
      "description": ""
    }}
  ],
  "work_experience": []
}}

Details:
- "location" is the candidate's city or region.
- "experience_years" is the total years of professional experience (integer).
- "current_role" and "company" indicate their present job title and employer.
- "projects" is a list of projects with "title" and "description" fields.
- "education" and "work_experience" should list relevant details in arrays.

Resume text:
{resume_text}
"""


    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful resume parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    response.raise_for_status()  # Raise exception for HTTP errors

    content = response.json()["choices"][0]["message"]["content"]

    try:
        # Extract JSON object using regex
        match = re.search(r'{.*}', content, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in LLM response.")
        json_string = match.group(0)
        parsed_json = json.loads(json_string)
    except Exception as e:
        print("❌ Failed to parse LLM output. Raw content:")
        print(content)
        raise e
    print("Successfully parsed JSON: \n", parsed_json)
    return parsed_json