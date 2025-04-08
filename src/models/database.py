from sqlalchemy import create_engine, Column, String, DateTime, Integer, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from datetime import datetime
import os

Base = declarative_base()

class KYCRecord(Base):
    __tablename__ = 'kyc_records'

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True)
    full_name = Column(String)
    document_type = Column(String)
    document_number = Column(String)
    date_of_birth = Column(DateTime, nullable=True)
    address = Column(String)
    verification_status = Column(String, default='pending')
    document_path = Column(String)
    selfie_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create SQLite database engine with connection pooling
DATABASE_URL = "sqlite:///./kyc.db"
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# Create database directory if it doesn't exist
os.makedirs(os.path.dirname(DATABASE_URL.replace('sqlite:///', '')), exist_ok=True)

# Create all tables
Base.metadata.create_all(bind=engine)

# Create scoped session factory
session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionLocal = scoped_session(session_factory)

@event.listens_for(engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA synchronous=NORMAL')
    cursor.execute('PRAGMA temp_store=MEMORY')
    cursor.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()