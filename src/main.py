from fastapi import FastAPI
from db.db_engine import DBEngine
from db.base import Base
from router import router

dbeng = DBEngine()
Base.metadata.create_all(bind=dbeng.engine)
app = FastAPI()
app.include_router(router)