from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Integer, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"
    
    email = Column(String, primary_key=True)
    name = Column(String)
    picture = Column(String)

    # Relationships
    engagements = relationship("Engagement", back_populates="user")
    files = relationship("UploadedFile", back_populates="user", foreign_keys="UploadedFile.user_email")

class Engagement(Base):
    __tablename__ = "engagements"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.email"))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="engagements")
    files = relationship("UploadedFile", back_populates="engagement")
    scan_results = relationship("ScanResult", back_populates="engagement")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(String, primary_key=True)
    original_filename = Column(String)
    stored_filename = Column(String)
    user_email = Column(String, ForeignKey('users.email'))
    upload_date = Column(DateTime, default=datetime.utcnow)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    engagement_id = Column(String, ForeignKey('engagements.id'))
    
    # Relationships
    user = relationship("User", back_populates="files", foreign_keys=[user_email])
    engagement = relationship("Engagement", back_populates="files")

class Incidents(Base):
    __tablename__ = "incidents"

    number = Column(String, primary_key=True)
    issue_description = Column(String)
    reassignment_count = Column(Integer)
    reopen_count = Column(Integer)
    made_sla = Column(Boolean)
    caller_id = Column(String)
    opened_at = Column(DateTime)
    category = Column(String)
    subcategory = Column(String)
    u_symptom = Column(String)
    cmdb_ci = Column(String)
    priority = Column(String)
    assigned_to = Column(String)
    problem_id = Column(String)
    resolved_by = Column(String)
    closed_at = Column(DateTime)
    resolved_at = Column(DateTime)
    resolution_notes = Column(String)
    resolution_time = Column(Float)
    complexity = Column(String)
    # Add new score fields
    customer_satisfaction = Column(String)
    resolution_time_score = Column(String)
    reassignment_score = Column(String)
    reopen_score = Column(String)
    sentiment_score = Column(String)
    sentiment = Column(String)

class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    engagement_id = Column(String, ForeignKey('engagements.id'))
    repository_url = Column(String)
    results = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    engagement = relationship("Engagement", back_populates="scan_results")
