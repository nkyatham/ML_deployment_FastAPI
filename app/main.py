from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import model_predict, __version__ as model_version


app = FastAPI()

class TextIn(BaseModel):
    Id: str
    AB: float 
    AF: float 
    AH: float 
    AM: float 
    AR: float 
    AX: float 
    AY: float 
    AZ: float 
    BC: float 
    BD: float 
    BN: float
    BP: float 
    BQ: float 
    BR: float 
    BZ: float 
    CB: float 
    CC: float 
    CD: float 
    CF: float 
    CH: float 
    CL: float 
    CR: float 
    CS: float
    CU: float 
    CW: float 
    DA: float 
    DE: float 
    DF: float 
    DH: float 
    DI: float 
    DL: float 
    DN: float 
    DU: float 
    DV: float 
    DY: float
    EB: float 
    EE: float 
    EG: float 
    EH: float 
    EJ: str
    EL: float
    EP: float
    EU: float
    FC: float
    FD: float
    FE: float
    FI: float
    FL: float
    FR: float
    FS: float
    GB: float
    GE: float
    GF: float
    GH: float
    GI: float
    GL: float

class PredictionOut(BaseModel):
    Probability: float


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    prediction_probability = model_predict(payload.dict())
    return {"Probability": prediction_probability}