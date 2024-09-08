from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional, Dict
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from database import SessionLocal, engine, get_db
import models
import requests
import csv
import os
from io import StringIO
from dateutil import parser as date_parser
import pandas as pd
import uuid
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import sqlalchemy
from sqlalchemy import func, desc
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sqlalchemy import text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    resolution_notes: str
    resolution_time: float
    complexity: str

class IncidentModel(IncidentBase):
    number: str = Field(alias='incident_number')
    complexity: str

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


class MappingData(BaseModel):
    mapping: Dict[str, str]
    resolution_time_field: str

@app.post("/mapping")
async def submit_mapping(
    mapping_data: MappingData,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Here you would typically process the mapping and resolution time field
    # For example, you might store it in the database or use it to process the uploaded file
    
    return {"message": "Mapping received successfully"}

@app.post("/process")
async def process_data(
    mapping_data: MappingData,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        logger.info("Authentication successful")

        # Get the latest uploaded file for the user
        latest_file = db.query(models.UploadedFile).filter(
            models.UploadedFile.user_email == current_user.email
        ).order_by(models.UploadedFile.upload_date.desc()).first()

        if not latest_file:
            raise HTTPException(status_code=404, detail="No uploaded file found")

        file_path = f"uploads/{latest_file.stored_filename}"
        logger.info(f"Processing file: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"CSV file read successfully. Shape: {df.shape}")

        # Apply mappings and handle data types
        mapped_df = pd.DataFrame()
        for internal_field, csv_field in mapping_data.mapping.items():
            if csv_field in df.columns:
                if internal_field in ['reassignment_count', 'reopen_count']:
                    mapped_df[internal_field] = pd.to_numeric(df[csv_field], errors='coerce').fillna(0).astype(int)
                elif internal_field == 'made_sla':
                    # Convert 'TRUE' and 'FALSE' strings to boolean values
                    mapped_df[internal_field] = df[csv_field].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    # Fill any remaining NaN values with False
                    mapped_df[internal_field] = mapped_df[internal_field].fillna(False)
                else:
                    mapped_df[internal_field] = df[csv_field]
            else:
                logger.warning(f"Column {csv_field} not found in CSV file")
        logger.info("Mappings applied")

        # Convert datetime fields
        datetime_fields = ['opened_at', 'resolved_at', 'closed_at']
        for field in datetime_fields:
            if field in mapped_df.columns:
                mapped_df[field] = pd.to_datetime(mapped_df[field], format='%d/%m/%Y %H:%M', errors='coerce')
            else:
                logger.warning(f"Datetime field {field} not found in DataFrame")
        logger.info("Datetime fields converted")

        # Calculate resolution time
        if 'opened_at' in mapped_df.columns and 'resolved_at' in mapped_df.columns and 'closed_at' in mapped_df.columns:
            mapped_df['resolution_time'] = (mapped_df['resolved_at'] - mapped_df['opened_at']).dt.total_seconds()
            # If resolution_time is negative or zero, use closed_at instead
            mask = (mapped_df['resolution_time'] <= 0) | (mapped_df['resolution_time'].isna())
            mapped_df.loc[mask, 'resolution_time'] = (mapped_df.loc[mask, 'closed_at'] - mapped_df.loc[mask, 'opened_at']).dt.total_seconds()
            # Ensure resolution_time is always positive
            mapped_df['resolution_time'] = mapped_df['resolution_time'].clip(lower=1)
            logger.info("Resolution time calculated")
        else:
            logger.warning("Could not calculate resolution time due to missing columns")

        # Dynamic split point adjustment for complexity classification
        if 'resolution_time' in mapped_df.columns and 'reassignment_count' in mapped_df.columns:
            scaler = StandardScaler()
            normalized_resolution_time = scaler.fit_transform(mapped_df[['resolution_time']])
            normalized_reassignment_count = scaler.fit_transform(mapped_df[['reassignment_count']])

            def calculate_complexity(res_time, reassign_count):
                if pd.isna(res_time) or pd.isna(reassign_count):
                    return 'Unknown'
                score = res_time + reassign_count
                if score < -1:
                    return 'Simple'
                elif score < 0:
                    return 'Medium'
                elif score < 1:
                    return 'Hard'
                else:
                    return 'Complex'

            mapped_df['complexity'] = [calculate_complexity(rt, rc) for rt, rc in zip(normalized_resolution_time, normalized_reassignment_count)]
            logger.info("Complexity calculated with dynamic split points")
        else:
            logger.warning("Could not calculate complexity due to missing columns")

        # Remove duplicate entries based on the 'number' field
        mapped_df = mapped_df.drop_duplicates(subset=['number'], keep='first')
        logger.info(f"Removed duplicate entries. New shape: {mapped_df.shape}")

        # Create a unique table name for the user
        table_name = f"incident_data_{current_user.id}"

        # Create SQLAlchemy Table dynamically
        from sqlalchemy import Table, MetaData, Column, String, Integer, Float, DateTime, Boolean
        metadata = MetaData()
        incident_table = Table(table_name, metadata,
            Column('number', String, primary_key=True),
            Column('issue_description', String),
            Column('reassignment_count', Integer),
            Column('reopen_count', Integer),
            Column('made_sla', Boolean),
            Column('caller_id', String),
            Column('opened_at', DateTime),
            Column('category', String),
            Column('subcategory', String),
            Column('u_symptom', String),
            Column('cmdb_ci', String),
            Column('priority', String),
            Column('assigned_to', String),
            Column('problem_id', String),
            Column('resolved_by', String),
            Column('closed_at', DateTime),
            Column('resolved_at', DateTime),
            Column('resolution_notes', String),
            Column('resolution_time', Float),
            Column('complexity', String)
        )

        # Create the table in the database
        engine = create_engine(f'sqlite:///./zinc_{current_user.id}.db')
        metadata.create_all(engine)

        # Insert data into the database
        with engine.connect() as connection:
            for _, row in mapped_df.iterrows():
                row_dict = row.to_dict()
                # Ensure boolean values are properly handled
                for key, value in row_dict.items():
                    if pd.isna(value):
                        if key == 'made_sla':
                            row_dict[key] = False
                        elif isinstance(value, (int, float)):
                            row_dict[key] = 0
                        else:
                            row_dict[key] = None
                try:
                    insert_stmt = incident_table.insert().values(**row_dict)
                    connection.execute(insert_stmt)
                except sqlalchemy.exc.IntegrityError as e:
                    # logger.warning(f"Duplicate entry found for number: {row_dict['number']}. Skipping.")
                    continue
            connection.commit()

        logger.info(f"Data inserted into the database table: {table_name}")

        return {"message": "Data processed and inserted into the database successfully"}

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/report/{user_id}")
async def get_report(user_id: int, db: Session = Depends(get_db)):
    try:
        engine = create_engine(f'sqlite:///./zinc_{user_id}.db')
        logger.info(f"Created engine for user {user_id}")
        
        with engine.connect() as connection:
            table_name = f"incident_data_{user_id}"
            logger.info(f"Querying table: {table_name}")
            
            def execute_query(query, description):
                logger.info(f"Executing query: {description}")
                result = connection.execute(text(query))
                logger.info(f"Query executed: {description}")
                return result.first()

            most_complex = execute_query(f"""
                SELECT category, subcategory, u_symptom, COUNT(*) as count
                FROM {table_name}
                WHERE complexity = 'Complex'
                GROUP BY category, subcategory, u_symptom
                ORDER BY count DESC
                LIMIT 1
            """, "Most complex incident type")

            date_range = execute_query(f"""
                SELECT MIN(opened_at) as min_date, MAX(opened_at) as max_date
                FROM {table_name}
            """, "Date range")

            avg_resolution_time = execute_query(f"""
                SELECT AVG(resolution_time) / 60 as avg_resolution_time
                FROM {table_name}
            """, "Average resolution time")

            logger.info("Executing table data query")
            table_data = connection.execute(text(f"""
                SELECT category, subcategory, u_symptom, 
                       COUNT(*) as incident_count,
                       AVG(resolution_time) / 60 as avg_resolution_time,
                       MAX(issue_description) as issue_description,
                       MAX(resolution_notes) as resolution_notes
                FROM {table_name}
                GROUP BY category, subcategory, u_symptom
                ORDER BY incident_count DESC
            """)).fetchall()
            logger.info(f"Table data query executed, fetched {len(table_data)} rows")

        def safe_dict(row):
            if row is None:
                return None
            return {column: getattr(row, column) for column in row._fields}

        report_data = {
            "cards": {
                "most_complex": safe_dict(most_complex),
                "date_range": safe_dict(date_range),
                "avg_resolution_time": safe_dict(avg_resolution_time)
            },
            "table_data": [safe_dict(row) for row in table_data]
        }

        logger.info("Report data compiled successfully")
        return report_data

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/generate_sop")
async def generate_sop(data: dict):
    try:
        issue_description = data.get("issue_description")
        resolution_notes = data.get("resolution_notes")

        llm = Ollama(model="llama3.1")
        
        template = """
        Given the following incident description and resolution notes, generate a detailed Standard Operating Procedure (SOP) to prevent similar incidents in the future and proactively address potential issues:

        Incident Description: {issue_description}

        Resolution Notes: {resolution_notes}

        Please provide a comprehensive SOP that includes:
        1. Steps to prevent this type of incident from occurring again
        2. Proactive measures to identify and address similar issues before they happen
        3. Best practices for handling this type of incident if it does occur
        4. Any necessary training or knowledge sharing to prevent future occurrences

        SOP:
        """

        prompt = PromptTemplate(template=template, input_variables=["issue_description", "resolution_notes"])
        chain = LLMChain(llm=llm, prompt=prompt)

        sop = chain.run(issue_description=issue_description, resolution_notes=resolution_notes)

        return {"sop": sop}
    except Exception as e:
        logger.error(f"Error generating SOP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SOP: {str(e)}")

