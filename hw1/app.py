import joblib
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


app = FastAPI()


def load_model() -> Pipeline:
    model = joblib.load("model_dump.pickle")
    return model

class Item(BaseModel):
    name: str
    year: int
    selling_price: Optional[float] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = None


class Items(BaseModel):
    objects: List[Item]


@app.get("/")
def root():
    return {"message": "HW1 ML HSE Saraev Nikita"}


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_request = pd.DataFrame([item.model_dump()])
    df_request = df_request.drop(columns="selling_price")
    model = load_model()
    prediction = model.predict(df_request)[0]
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> Items:
    records = list(map(lambda x: x.model_dump(), items))
    df_request = pd.DataFrame(records)
    if "selling_price" in df_request.columns:
        df_request = df_request.drop(columns="selling_price")
    model = load_model()
    df_request["selling_price"] = model.predict(df_request)
    response_records = df_request.to_dict(orient="records")
    response = {"objects": response_records}
    return response
