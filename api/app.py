import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import run_inference

app = FastAPI()

class SummaryRequest(BaseModel):
    medical_text: str

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/summarize")
def summarize(request: SummaryRequest):  
    summary = run_inference(request.medical_text)
    return {
        "input": request.medical_text,
        "summary": summary
        }