
import io
import os
from fastapi import APIRouter, Form, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi import File, UploadFile

from bayes import Bayes


router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def view_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "name": " & Welcome Back To Bayes !"})

@router.get("/upload", response_class=HTMLResponse)
async def view_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

UPLOAD_DIR = "upload"
@router.post("/post-upload")
async def upload_data(file: UploadFile = File(...)):
        try:
            if file.content_type != "text/csv":
                raise HTTPException(status_code=400, detail="File type not supported")
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)

            # Menyimpan file ke direktori yang ditentukan
            with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
                contents = await file.read()  # baca file
                buffer.write(contents)

            files = []
            if os.path.exists(UPLOAD_DIR):
                files = os.listdir(UPLOAD_DIR)
            
            return {
                "error": "",
                "status": True,
                "data": files
            }
    
        except HTTPException as e:
            return {
                "error": e.detail,
                "status": False,
                "data": []
            }
        
@router.get("/train", response_class=HTMLResponse)
async def view_train(request: Request):
    files = None
    if os.path.exists(UPLOAD_DIR):
        files = os.listdir(UPLOAD_DIR)
    return templates.TemplateResponse("train.html", {"request": request, "files":files, "error":""})

@router.post("/start-training", response_class=HTMLResponse)
async def start_training(
    request: Request,
    selectedFile: str = Form(...),
    splitPercentage: float = Form(...)
):
    filename = f'./upload/{selectedFile}'
    df = pd.read_csv(filename)
    nb = Bayes()
    result = nb.start(df, splitPercentage)
    return templates.TemplateResponse("results.html", 
            {"request": request,
            "metrics": result['metrics'],
            "plot_url": f"/static/{result['plot_filename']}",
            })

@router.get("/predict", response_class=HTMLResponse)
async def start_test(
    request: Request,
    # tgl_faktur: str = Form(...),
    # nik: str = Form(...),
    # alamat: str = Form(...)
):
    return templates.TemplateResponse("predict.html", {"request": request, "error":"", "success":""})

@router.post("/start-predict", response_class=HTMLResponse)
async def start_predict(
    request: Request,
    tgl_faktur: str = Form(...),
    nik: str = Form(...),
    alamat: str = Form(...)
):
    print(tgl_faktur, nik, alamat)
    input_data = {
    'tgl_faktur': tgl_faktur,
    'NIK': nik,
    'Alamat': alamat
}
    input_df = pd.DataFrame(input_data, index=[0])
    nb = Bayes()
    predict = nb.predict(input_df)
    return templates.TemplateResponse("predict.html", {"request": request, "error":"", "success":f'Hasil prediksinya adalah {predict}'})
