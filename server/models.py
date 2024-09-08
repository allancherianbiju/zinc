from database import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

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
