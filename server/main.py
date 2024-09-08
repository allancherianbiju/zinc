from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional, Dict
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from database import SessionLocal, engine
import models
import requests
import csv
import os
from io import StringIO
from dateutil import parser as date_parser
import pandas as pd
import uuid

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IncidentBase(BaseModel):
    number: str
    issue_description: str
    reassignment_count: int
    reopen_count: int
    made_sla: bool
    caller_id: str
    opened_at: datetime
    category: str
    subcategory: str
    u_symptom: str
    cmdb_ci: str
    priority: str
    assigned_to: str
    problem_id: str
    resolved_by: str
    closed_at: datetime
    resolved_at: datetime
    resolution_notes: str  # New field
    resolution_time: int  # New field (in seconds)

class IncidentModel(IncidentBase):
    number: str = Field(alias='incident_number')

    class Config:
        orm_mode = True

class GoogleToken(BaseModel):
    token: str

class User(BaseModel):
    email: str
    name: str
    picture: str

class UploadedFile(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    user_email: str
    upload_date: datetime

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

models.Base.metadata.create_all(bind=engine)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    user = db.query(models.User).filter(models.User.email == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

@app.post("/incidents/", response_model=IncidentModel)
async def create_incident(incident: IncidentBase, db: db_dependency):
    db_incident = models.Incidents(**incident.model_dump(by_alias=True))
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)
    return db_incident

@app.post("/auth/google")
async def google_auth(token: GoogleToken, db: Session = Depends(get_db)):
    try:
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token.token}"}
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()

        user = models.User(
            email=userinfo['email'],
            name=userinfo['name'],
            picture=userinfo['picture']
        )

        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if not db_user:
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user = db_user

        return {"user": user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/logout")
async def logout():
    return {"message": "Logged out successfully"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user: Optional[models.User] = Depends(get_current_user)
):
    if not current_user:
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})

    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"detail": "Only CSV files are allowed"})

    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = f"uploads/{unique_filename}"

    # Check if a file with the same original name exists for this user
    existing_file = db.query(models.UploadedFile).filter(
        models.UploadedFile.user_email == current_user.email,
        models.UploadedFile.original_filename == file.filename
    ).first()

    if existing_file:
        # Remove the old file
        old_file_path = f"uploads/{existing_file.stored_filename}"
        if os.path.exists(old_file_path):
            os.remove(old_file_path)
        
        # Update the database entry
        existing_file.stored_filename = unique_filename
        existing_file.upload_date = datetime.now()
    else:
        # Create a new database entry
        new_file = models.UploadedFile(
            id=str(uuid.uuid4()),
            original_filename=file.filename,
            stored_filename=unique_filename,
            user_email=current_user.email,
            upload_date=datetime.now()
        )
        db.add(new_file)

    # Save the new file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Get the shape of the data
    rows, columns = df.shape

    # Get a preview of the data (first 10 rows)
    preview_data = df.head(10).to_dict('records')

    # Commit the changes to the database
    db.commit()

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "rows": rows,
        "columns": columns,
        "preview_data": preview_data,
    }

def parse_datetime(date_string):
    if pd.isna(date_string):
        return None
    return date_parser.parse(date_string)


@app.post("/mapping")
async def submit_mapping(
    mapping: Dict[str, str] = Body(...),
    resolution_time_field: str = Body(...),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Here you would typically process the mapping and resolution time field
    # For example, you might store it in the database or use it to process the uploaded file
    
    return {"message": "Mapping received successfully"}

