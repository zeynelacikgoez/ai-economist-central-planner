from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class Experience(Base):
    __tablename__ = 'experiences'
    id = Column(Integer, primary_key=True, autoincrement=True)
    observations = Column(JSON)
    actions = Column(JSON)
    rewards = Column(Float)
    next_observations = Column(JSON)

# Datenbank-Setup
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///economy.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initialisiert die Datenbank und erstellt die Tabellen.
    """
    Base.metadata.create_all(bind=engine)
