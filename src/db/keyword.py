from sqlalchemy import Column, String
from sqlalchemy.orm import Session
from db.base import Base

class Keyword(Base):
    """
    SQLAlchemy model for the 'keywords' table.
    """
    __tablename__ = "keywords"
    keywords = Column(String, primary_key=True)

class KeywordRepository:
    """
    Repository class for handling database operations related to the 'keywords' table.
    """

    @staticmethod
    def get_all(db: Session):
        """
        Get all keywords from the 'keywords' table.

        Args:
            db (Session): SQLAlchemy database session.

        Returns:
            list[str]: List of keyword strings.
        """
        items = db.query(Keyword).all()
        return [item.keywords for item in items]

    @staticmethod
    def get_one(db: Session, name: str) -> Keyword:
        """
        Get a single keyword from the 'keywords' table by name.

        Args:
            db (Session): SQLAlchemy database session.
            name (str): Name of the keyword.

        Returns:
            Keyword object if found, else None.
        """
        result = db.query(Keyword).filter(Keyword.keywords == name).first()
        return result
