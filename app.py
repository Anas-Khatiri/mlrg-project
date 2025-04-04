from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
from typing import List, Dict, Union

from intern_src.utils_package.data_ingestion import load_pipeline
from intern_src.utils_package.data_prediction import validate_inputs
from configs import core
from intern_src.pipelines_package.predict import make_prediction

# Initialiser FastAPI
app = FastAPI()

# Définir le dossier des templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
def upload_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        data["MSSubClass"] = data["MSSubClass"].astype("O")
        # rename variables beginning with numbers to avoid syntax errors later
        data_transformed = data.rename(columns=core.VARIABLES_TO_RENAME)
        # Check the columns
        missing_cols = [col for col in core.FEATURES if col not in data_transformed.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing Columns: {missing_cols}")

        results = make_prediction(input_data=data_transformed[core.FEATURES].to_dict(orient='records'))
        return {"predictions": results["predictions"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error of file processing: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
