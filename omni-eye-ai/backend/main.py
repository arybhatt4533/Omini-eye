from fastapi import FastAPI
from routes import router

app = FastAPI(title="OmniEye Backend")

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Backend Running"}