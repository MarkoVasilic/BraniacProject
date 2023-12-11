from sqlalchemy import Column, String
from sqlalchemy.orm import Session
from db.base import Base

class ProductionCompany(Base):
    """
    SQLAlchemy model for the 'production_companies' table.
    """
    __tablename__ = "production_companies"
    production_companies = Column(String, primary_key=True)

class ProductionCompanyRepository:
    """
    Repository class for handling database operations related to the 'production_companies' table.
    """

    @staticmethod
    def get_all(db: Session):
        """
        Get all production companies from the 'production_companies' table.

        Args:
            db (Session): SQLAlchemy database session.

        Returns:
            List of production company strings.
        """
        items = db.query(ProductionCompany).all()
        return [item.production_companies for item in items]

    @staticmethod
    def get_one(db: Session, name: str) -> ProductionCompany:
        """
        Get a single production company from the 'production_companies' table by name.

        Args:
            db (Session): SQLAlchemy database session.
            name (str): Name of the production company.

        Returns:
            ProductionCompany: ProductionCompany object if found, else None.
        """
        result = db.query(ProductionCompany).filter(ProductionCompany.production_companies == name).first()
        return result
