from database import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship

class Incidents(Base):
    __tablename__ = "incident_data"
    number = Column(String, primary_key=True, index=True)
    issue_description = Column(String)
    reassignment_count = Column(Integer)
    reopen_count = Column(Integer)
    made_sla = Column(Boolean)
    caller_id = Column(String, index=True)
    opened_at = Column(DateTime)
    category = Column(String)
    subcategory = Column(String)
    u_symptom = Column(String)
    cmdb_ci = Column(String)
    priority = Column(String)
    assigned_to = Column(String)
    problem_id = Column(String, index=True)
    resolved_by = Column(String)
    closed_at = Column(DateTime)
    resolved_at = Column(DateTime)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    picture = Column(String)
    uploaded_files = relationship("UploadedFile", back_populates="user")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(String, primary_key=True)
    original_filename = Column(String)
    stored_filename = Column(String)
    user_email = Column(String, ForeignKey('users.email'))
    upload_date = Column(DateTime)

    user = relationship("User", back_populates="uploaded_files")
