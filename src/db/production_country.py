from sqlalchemy import Column, String
from sqlalchemy.orm import Session
from db.base import Base

class ProductionCountry(Base):
    """
    SQLAlchemy model for the 'production_countries' table.
    """
    __tablename__ = "production_countries"
    production_countries = Column(String, primary_key=True)

class ProductionCountryRepository:
    """
    Repository class for handling database operations related to the 'production_countries' table.
    """

    @staticmethod
    def get_all(db: Session):
        """
        Get all production countries from the 'production_countries' table.

        Args:
            db (Session): SQLAlchemy database session.

        Returns:
            List of production country strings.
        """
        items = db.query(ProductionCountry).all()
        return [item.production_countries for item in items]

    @staticmethod
    def get_one(db: Session, name: str) -> ProductionCountry:
        """
        Get a single production country from the 'production_countries' table by name.

        Args:
            db (Session): SQLAlchemy database session.
            name (str): Name of the production country.

        Returns:
            ProductionCountry: ProductionCountry object if found, else None.
        """
        result = db.query(ProductionCountry).filter(ProductionCountry.production_countries == name).first()
        return result
