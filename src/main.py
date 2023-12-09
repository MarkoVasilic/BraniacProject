from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, MetaData, Table, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from db_engine import DBEngine
from typing import Optional

Base = declarative_base()

class ProductionCompany(Base):
    __tablename__ = "production_companies"
    production_companies = Column(String, primary_key=True)

class ProductionCountry(Base):
    __tablename__ = "production_countries"
    production_countries = Column(String, primary_key=True)

class Keyword(Base):
    __tablename__ = "keywords"
    keywords = Column(String, primary_key=True)

app = FastAPI()

# Create a FastAPI app
app = FastAPI()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize the database engine
dbeng = DBEngine()
Base.metadata.create_all(bind=dbeng.engine)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=dbeng.engine)

def get_all(table: Base, db, column_name):
    items = db.query(table).all()
    if column_name == 'production_companies':
        return [item.production_companies for item in items]
    if column_name == 'production_countries':
        return [item.production_countries for item in items]
    return [item.keywords for item in items]
    
def check_string_existence(table: Base, db: Session, filter_str: Optional[str] = None) -> bool:
    result = db.query(table).filter(table.production_countries == filter_str).all()
    return len(result) > 0

@app.get("/production_companies/", response_model=list[str])
def get_all_production_companies(db: Session = Depends(get_db)):
    return get_all(ProductionCompany, db, 'production_companies')

@app.get("/production_countries/", response_model=list[str])
def get_all_production_countries(db: Session = Depends(get_db)):
    return get_all(ProductionCountry, db, 'production_countries')

@app.get("/keywords/", response_model=list[str])
def get_all_keywords(db: Session = Depends(get_db)):
    return get_all(Keyword, db, 'keywords')

@app.get("/production_countries/check_existence", response_model=bool)
def check_production_countries_existence(filter_str: Optional[str] = None, db: Session = Depends(get_db)):
    return check_string_existence(ProductionCountry, db, filter_str)