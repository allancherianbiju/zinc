from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Integer, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

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
