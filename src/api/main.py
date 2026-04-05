from fastapi import FastAPI

app = FastAPI(title="Sketch-to-Face API")

@app.get("/")
def root():
    return {"status": "API running"}
