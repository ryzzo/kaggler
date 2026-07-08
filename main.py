import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import analysis, hyperparameter, inference, pages, preparation, resources, training, upload

app = FastAPI(title="File Processing API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(pages.router)
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(preparation.router)
app.include_router(training.router)
app.include_router(hyperparameter.router)
app.include_router(inference.router)
app.include_router(resources.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
