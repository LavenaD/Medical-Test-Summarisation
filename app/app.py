from flask import Flask, request, render_template_string
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Medical Text Summarization</title>
</head>
<body>
    <h1>Medical Text Summarization</h1>
    <form method="post">
        <label for="medical_text">Enter Medical Text:</label><br>
        <textarea id="medical_text" name="medical_text" rows="10" cols="50"></textarea><br><br>
        <input type="submit" value="Summarize">
    </form>
    {% if summary %}
        <h2>Summary:</h2>
        <p>{{ summary }}</p>
    {% endif %}
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        medical_text = request.form["medical_text"]
        response = requests.post(f"{os.getenv("API_URL")}/summarize", json={"medical_text": medical_text})
        summary = response.json().get("summary")
        print("Summary Response:", response.json())
    return render_template_string(html, summary=summary)

if __name__ == "__main__":
    app.run(debug=False)