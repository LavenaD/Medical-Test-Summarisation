from flask import Flask, render_template, request, render_template_string
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":

        medical_text = request.form["medical_text"]
        response = requests.post(f"{os.getenv("API_URL")}/summarize", json={"medical_text": medical_text})
        summary = response.json().get("summary")
        findings = f"Findings: {medical_text}"
        return render_template('index.html', findings=findings, summary=summary)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run( host='0.0.0.0', debug=False)