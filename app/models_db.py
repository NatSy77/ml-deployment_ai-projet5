from sqlalchemy import Column, BigInteger, Integer, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from app.db import Base

class Client(Base):
    __tablename__ = "clients"
    id = Column(BigInteger, primary_key=True, index=True)
    features = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class PredictionRequest(Base):
    __tablename__ = "prediction_requests"
    id = Column(BigInteger, primary_key=True, index=True)
    client_id = Column(BigInteger, ForeignKey("clients.id", ondelete="RESTRICT"), nullable=False, index=True)
    features = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class PredictionOutput(Base):
    __tablename__ = "prediction_outputs"
    id = Column(BigInteger, primary_key=True, index=True)
    request_id = Column(BigInteger, ForeignKey("prediction_requests.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(Integer, nullable=False)
    proba = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
