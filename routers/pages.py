"""Static page routes — each just serves its corresponding static/*.html."""
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/")
def index():
    return FileResponse("static/index.html")


@router.get("/analysis")
def analysis_page():
    return FileResponse("static/analysis.html")


@router.get("/preparation")
def preparation_page():
    return FileResponse("static/preparation.html")


@router.get("/training")
def training_page():
    return FileResponse("static/training.html")


@router.get("/inference")
def inference_page():
    return FileResponse("static/inference.html")
