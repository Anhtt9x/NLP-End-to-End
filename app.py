from src.pipeline.training_pipeline import Train_Pipeline
from fastapi import FastAPI
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception.exception import custom_exception
from src.constants import *

text:str = "What is machine learning"

app =  FastAPI()

@app.get("/",tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        train_pipeline = Train_Pipeline()
        train_pipeline.run_pipeline()

        return Response("Training successfull !")
    except Exception as e:
        return custom_exception(e,sys)
    
@app.get("/predict")
async def prediction():
    try:
        prediction_pipeline = PredictionPipeline()
        text=prediction_pipeline.run_pipeline()
        return text
    except Exception as e:
        return custom_exception(e,sys)
    

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)