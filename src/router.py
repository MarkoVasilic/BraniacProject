from fastapi import APIRouter
from fastapi import HTTPException, Depends
from db.db_engine import DBEngine
from sqlalchemy.orm import Session
import pandas as pd
from typing import List
from fastapi.responses import JSONResponse
from db.production_company import ProductionCompanyRepository
from db.production_country import ProductionCountryRepository
from db.keyword import KeywordRepository
import data.helper as helper
from models.models_dict import MODELS

router = APIRouter()

@router.get("/production_companies/")
def get_all_production_companies(db: Session = Depends(DBEngine().get_db)):
    return ProductionCompanyRepository.get_all(db)

@router.get("/production_countries/")
def get_all_production_countries(db: Session = Depends(DBEngine().get_db)):
    return ProductionCountryRepository.get_all(db)

@router.get("/keywords/")
def get_all_keywords(db: Session = Depends(DBEngine().get_db)):
    return KeywordRepository.get_all(db)

@router.get("/production_companies/{name}")
def get_one_production_companies(name : str, db: Session = Depends(DBEngine().get_db)):
    company = ProductionCompanyRepository.get_one(db, name)
    if company is None:
        raise HTTPException(status_code=404, detail="Production company not found.")
    return company

@router.get("/production_countries/{name}")
def get_one_production_countries(name : str, db: Session = Depends(DBEngine().get_db)):
    country = ProductionCountryRepository.get_one(db, name)
    if country is None:
        raise HTTPException(status_code=404, detail="Production country not found.")
    return country

@router.get("/keywords/{name}")
def get_one_keywords(name : str, db: Session = Depends(DBEngine().get_db)):
    keyword = KeywordRepository.get_one(db, name)
    if keyword is None:
        raise HTTPException(status_code=404, detail="Keyword not found.")
    return keyword

@router.post("/model/{model_name}/predict")
def predict(
    model_name: str = None,
    production_companies: List[str] = None,
    production_countries: List[str] = None,
    keywords: List[str] = None,
    db: Session = Depends(DBEngine().get_db)
):
    if model_name == None or (model_name != 'ml' and model_name != 'nn'):
        raise HTTPException(status_code=404, detail="Model name is wrong, it should be nn or ml.")
    
    if production_companies is None or production_countries is None or keywords is None:
        raise HTTPException(status_code=400, detail="All parameters must be provided.")
    
    for pc in production_companies:
        if ProductionCompanyRepository.get_one(db, pc) == None:
            raise HTTPException(status_code=404, detail=f"'{pc}' does not exist in the table production_company")

    for pc in production_countries:
        if ProductionCountryRepository.get_one(db, pc) == None:
            raise HTTPException(status_code=404, detail=f"'{pc}' does not exist in the table production_country")

    for k in keywords:
        if KeywordRepository.get_one(db, k) == None:
            raise HTTPException(status_code=404, detail=f"'{k}' does not exist in the table keyword")

    prod_comp = ProductionCompanyRepository.get_all(db)
    prod_coun = ProductionCountryRepository.get_all(db)
    prod_key = KeywordRepository.get_all(db)
    
    proccesed_data = helper.process_data_for_predict(prod_comp, prod_coun, prod_key, production_companies, production_countries, keywords)
    model = MODELS[model_name]
    model.load()
    result = model.predict([[proccesed_data[0], proccesed_data[1], proccesed_data[2]]])
    result = round(result, 2)
    return JSONResponse(content={"movie_rating": result})
