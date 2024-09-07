from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional
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

    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Print statistics and shape of the DataFrame
    print(f"\nFile uploaded: {file.filename}")
    print(f"Shape of the data: {df.shape}")
    print("\nDataFrame Statistics:")
    print(df.describe())
    print("\nColumn names:")
    print(df.columns.tolist())

    # Process the CSV file and insert data into the database
    rows_processed = 0
    try:
        for _, row in df.iterrows():
            incident_data = {
                "number": row['number'],
                "issue_description": row['issue_description'],
                "reassignment_count": int(row['reassignment_count']),
                "reopen_count": int(row['reopen_count']),
                "made_sla": row['made_sla'],  # Assuming this is already a boolean
                "caller_id": row['caller_id'],
                "opened_at": parse_datetime(row['opened_at']),
                "category": row['category'],
                "subcategory": row['subcategory'],
                "u_symptom": row['u_symptom'],
                "cmdb_ci": row['cmdb_ci'],
                "priority": row['priority'],
                "assigned_to": row['assigned_to'],
                "problem_id": row['problem_id'],
                "resolved_by": row['resolved_by'],
                "closed_at": parse_datetime(row['closed_at']),
                "resolved_at": parse_datetime(row['resolved_at']),
            }
            db_incident = models.Incidents(**incident_data)
            db.add(db_incident)
            rows_processed += 1

        # Record the uploaded file
        uploaded_file = models.UploadedFile(
            id=str(uuid.uuid4()),
            original_filename=file.filename,
            stored_filename=unique_filename,
            user_email=current_user.email,
            upload_date=datetime.now()
        )
        db.add(uploaded_file)

        db.commit()
        print(f"\nDatabase operation completed successfully. {rows_processed} rows processed.")
    except Exception as e:
        db.rollback()
        print(f"\nError during database operation: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Error processing file: {str(e)}"})

    return {"message": "File uploaded and processed successfully", "filename": file.filename, "rows_processed": rows_processed}

def parse_datetime(date_string):
    if pd.isna(date_string):
        return None
    return date_parser.parse(date_string)

