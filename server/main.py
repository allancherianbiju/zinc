from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Body, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional, Dict, Any
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
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sqlalchemy import text
from werkzeug.utils import secure_filename
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import xml.etree.ElementTree as ET
import json
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import time
from starlette.responses import StreamingResponse
import asyncio

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
    customer_satisfaction: Optional[str] = None
    resolution_time_score: Optional[str] = None
    reassignment_score: Optional[str] = None
    reopen_score: Optional[str] = None
    sentiment_score: Optional[str] = None
    sentiment: Optional[str] = None

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
        logger.warning("No authentication token provided")
        return None
    
    # Add debug logging
    logger.info(f"Attempting to authenticate user with token: {token}")
    
    user = db.query(models.User).filter(models.User.email == token).first()
    if not user:
        logger.warning(f"No user found for token: {token}")
        return None
    
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
    engagement_type: str = Form(...),
    engagement_name: Optional[str] = Form(default=None),
    engagement_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    # Add debug logging
    logger.info(f"Received upload request: engagement_type={engagement_type}, "
                f"engagement_name={engagement_name}, engagement_id={engagement_id}")
    
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    if engagement_type not in ["new", "existing"]:
        raise HTTPException(status_code=400, detail="Invalid engagement type")

    if engagement_type == "new":
        if not engagement_name:
            raise HTTPException(status_code=400, detail="Engagement name is required for new engagements")
        
        # Check if engagement name already exists for this user
        existing_engagement = db.query(models.Engagement).filter(
            models.Engagement.user_id == current_user.email,
            models.Engagement.name == engagement_name
        ).first()
        
        if existing_engagement:
            raise HTTPException(status_code=400, detail="Engagement name already exists")
        
        engagement = models.Engagement(
            id=str(uuid.uuid4()),
            name=engagement_name,
            user_id=current_user.email
        )
        db.add(engagement)
        db.commit()  # Commit here to ensure engagement is created
        
    else:  # existing engagement
        if not engagement_id:
            raise HTTPException(status_code=400, detail="Engagement ID is required for existing engagements")
        
        engagement = db.query(models.Engagement).filter(
            models.Engagement.id == engagement_id,
            models.Engagement.user_id == current_user.email
        ).first()
        
        if not engagement:
            raise HTTPException(status_code=404, detail="Engagement not found")

    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Generate unique filename and save file
    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    file_path = f"uploads/{unique_filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Read CSV to determine date range
    df = pd.read_csv(file_path)
    if 'opened_at' in df.columns:
        date_range_start = pd.to_datetime(df['opened_at'].min())
        date_range_end = pd.to_datetime(df['opened_at'].max())
    else:
        date_range_start = None
        date_range_end = None

    # Create file record
    uploaded_file = models.UploadedFile(
        id=str(uuid.uuid4()),
        original_filename=file.filename,
        stored_filename=unique_filename,
        user_email=current_user.email,
        engagement_id=engagement.id,
        date_range_start=date_range_start,
        date_range_end=date_range_end
    )
    
    db.add(uploaded_file)
    db.commit()

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "preview_data": df.head(10).to_dict('records'),
        "engagement": {
            "id": engagement.id,
            "name": engagement.name
        },
        "account": engagement.name
    }

def parse_datetime(date_string):
    if pd.isna(date_string):
        return None
    try:
        return pd.to_datetime(date_string, format='%d-%m-%Y %H:%M:%S').to_pydatetime()
    except:
        try:
            # Fallback to more flexible parsing if exact format fails
            return date_parser.parse(date_string, dayfirst=True)
        except:
            return None


class MappingData(BaseModel):
    mapping: Dict[str, str]

@app.route('/preview', methods=['POST'])
def preview_file():
    if 'file' not in request.files:
        return JSONResponse(status_code=400, content={"detail": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return JSONResponse(status_code=400, content={"detail": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        
        preview_data = df.head(10).to_dict('records')
        
        empty_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        return JSONResponse({
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "preview_data": preview_data,
            "empty_percentages": empty_percentages
        }, status_code=200)
    return JSONResponse(status_code=400, content={"detail": "File type not allowed"})

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

# Add this near the top with other constants
datetime_fields = ['opened_at', 'closed_at', 'resolved_at']

# Add this field mapping dictionary at the top level of the file
FIELD_MAPPING = {
    "Number": "number",
    "Issue Description": "issue_description",
    "Reassignment Count": "reassignment_count",
    "Reopen Count": "reopen_count",
    "Made SLA": "made_sla",
    "Caller ID": "caller_id",
    "Opened At": "opened_at",
    "Category": "category",
    "Subcategory": "subcategory",
    "Symptom": "u_symptom",
    "Confirmation Item": "cmdb_ci",
    "Priority": "priority",
    "Assigned To": "assigned_to",
    "Problem ID": "problem_id",
    "Resolved By": "resolved_by",
    "Closed At": "closed_at",
    "Resolved At": "resolved_at",
    "Resolution Notes": "resolution_notes"
}

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

def remove_stopwords(text: str) -> str:
    """Remove stopwords to reduce token count"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
    return filtered_text

async def analyze_sentiments_batch(df: pd.DataFrame) -> dict:
    """Analyze sentiments for incident resolution notes asynchronously"""
    try:
        results = {}
        df['filtered_notes'] = df['resolution_notes'].apply(remove_stopwords)
        logger.info(f"Starting sentiment analysis for {len(df)} records")
        
        try:
            llm = Ollama(
                # model="llama3.1",
                model="phi3.5",
                temperature=0.1,
                num_gpu=1,
                system="You are a sentiment analyzer. Respond with ONLY ONE WORD from these options: highly_positive, positive, neutral, negative, highly_negative."
            )
            logger.info("Successfully initialized Ollama LLM")
        except Exception as llm_error:
            logger.error(f"Failed to initialize Ollama LLM: {str(llm_error)}")
            return {}

        chunk_size = 50
        
        async def process_row(row):
            try:
                number = str(row['number'])
                notes = row['filtered_notes']
                
                if pd.isna(notes) or not str(notes).strip():
                    return number, 'neutral'
                
                prompt = f"""Analyze this IT support ticket resolution note sentiment. Respond with ONLY ONE WORD from: highly_positive, positive, neutral, negative, highly_negative.

                Note: {notes}

                Sentiment:"""
                
                try:
                    response = await llm.ainvoke(prompt)
                    sentiment = str(response).strip().lower()
                    
                    # Map underscores to spaces and validate
                    sentiment = sentiment.replace('_', ' ')
                    valid_sentiments = {'highly positive', 'positive', 'neutral', 'negative', 'highly negative'}
                    
                    if sentiment not in valid_sentiments:
                        logger.warning(f"Invalid sentiment '{sentiment}' for ticket {number}, defaulting to neutral")
                        sentiment = 'neutral'
                    
                    return number, sentiment
                    
                except Exception as msg_error:
                    logger.error(f"Error processing message for ticket {number}: {str(msg_error)}")
                    return number, 'neutral'
                
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                return number if 'number' in locals() else 'unknown', 'neutral'

        # Process in chunks
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_number = (i // chunk_size) + 1
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            
            logger.info(f"Processing chunk {chunk_number}/{total_chunks}")
            
            try:
                tasks = [process_row(row) for _, row in chunk_df.iterrows()]
                chunk_results = await asyncio.gather(*tasks)
                results.update(dict(chunk_results))
                logger.info(f"Completed chunk {chunk_number}/{total_chunks}")
                
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {chunk_number}: {str(chunk_error)}")
                continue
            
        logger.info(f"Sentiment analysis completed. Processed {len(results)} tickets successfully.")
        return results
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {}

async def process_file_data(df: pd.DataFrame, mapping_data: dict) -> pd.DataFrame:
    """Process the uploaded file data with mappings"""
    # Convert frontend field names to internal names
    internal_mapping = {}
    for frontend_field, csv_column in mapping_data.items():
        internal_field = FIELD_MAPPING.get(frontend_field)
        if internal_field:
            internal_mapping[internal_field] = csv_column
        else:
            logger.warning(f"Unknown frontend field name: {frontend_field}")

    # Apply mappings and handle data types
    mapped_df = pd.DataFrame()
    for internal_field, csv_field in internal_mapping.items():
        if csv_field in df.columns:
            if internal_field in ['reassignment_count', 'reopen_count']:
                mapped_df[internal_field] = pd.to_numeric(df[csv_field], errors='coerce').fillna(0).astype(int)
            elif internal_field == 'made_sla':
                mapped_df[internal_field] = df[csv_field].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                mapped_df[internal_field] = mapped_df[internal_field].fillna(False)
            else:
                mapped_df[internal_field] = df[csv_field]
    
    return mapped_df

@app.post("/process")
async def process_data(
    mapping_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    try:
        processing_complete = False

        async def event_stream():
            nonlocal processing_complete
            if processing_complete:
                return
                
            sent_messages = set()
            completed_steps = set()
            
            try:
                # Updated progress steps with ETA
                progress_steps = {
                    "file_reading": {
                        "message": "Reading and validating file...",
                        "progress": 10,
                        "order": 1,
                        "eta": "Calculating..."
                    },
                    "data_mapping": {
                        "message": "Mapping data fields...",
                        "progress": 20,
                        "order": 2,
                        "eta": "Calculating..."
                    },
                    "datetime_processing": {
                        "message": "Processing timestamps...",
                        "progress": 30,
                        "order": 3,
                        "eta": "Calculating..."
                    },
                    "sentiment_analysis": {
                        "message": "Analyzing sentiments...",
                        "progress": 60,
                        "order": 4,
                        "eta": "Calculating..."
                    },
                    "scoring_calculation": {
                        "message": "Calculating scores...",
                        "progress": 80,
                        "order": 5,
                        "eta": "Calculating..."
                    },
                    "database_saving": {
                        "message": "Saving results...",
                        "progress": 100,
                        "order": 6,
                        "eta": "Almost done..."
                    }
                }

                # Get the latest uploaded file
                latest_file = db.query(models.UploadedFile).filter(
                    models.UploadedFile.user_email == current_user.email
                ).order_by(models.UploadedFile.upload_date.desc()).first()

                if not latest_file:
                    raise HTTPException(status_code=404, detail="No uploaded file found")

                file_path = f"uploads/{latest_file.stored_filename}"
                
                # Execute steps only if not already completed
                if "file_reading" not in completed_steps:
                    completed_steps.add("file_reading")
                    yield json.dumps(progress_steps["file_reading"]) + "\n"
                    df = pd.read_csv(file_path)
                    await asyncio.sleep(0.1)  # Small delay to ensure proper streaming

                # Data Mapping
                if "data_mapping" not in completed_steps:
                    completed_steps.add("data_mapping")
                    yield json.dumps(progress_steps["data_mapping"]) + "\n"
                    mapped_df = await process_file_data(df, mapping_data['mapping'])

                # DateTime Processing
                if "datetime_processing" not in completed_steps:
                    completed_steps.add("datetime_processing")
                    yield json.dumps(progress_steps["datetime_processing"]) + "\n"
                    
                    # Verify datetime consistency and convert
                    for field in datetime_fields:
                        if field in mapped_df.columns:
                            invalid_dates = mapped_df[field].isna()
                            if invalid_dates.any():
                                logger.warning(f"Found {invalid_dates.sum()} invalid dates in {field}")
                            
                            mapped_df[field] = pd.to_datetime(
                                mapped_df[field], 
                                format='%d/%m/%Y %H:%M',
                                dayfirst=True,
                                errors='coerce'
                            )

                    # Calculate resolution time
                    if 'opened_at' in mapped_df.columns and 'closed_at' in mapped_df.columns:
                        resolution_time = (mapped_df['closed_at'] - mapped_df['opened_at']).dt.total_seconds()
                        mapped_df['resolution_time'] = resolution_time

                # Sentiment Analysis
                if "sentiment_analysis" not in completed_steps:
                    completed_steps.add("sentiment_analysis")
                    yield json.dumps(progress_steps["sentiment_analysis"]) + "\n"
                    
                    # Await the sentiment analysis results
                    sentiments = await analyze_sentiments_batch(mapped_df)
                    mapped_df['sentiment'] = mapped_df['number'].map(lambda x: sentiments.get(x, 'neutral'))
                    mapped_df['sentiment'] = mapped_df['sentiment'].fillna('neutral')

                # Scoring Calculation
                if "scoring_calculation" not in completed_steps:
                    completed_steps.add("scoring_calculation")
                    yield json.dumps(progress_steps["scoring_calculation"]) + "\n"
                    
                    # Complexity calculation
                    if 'resolution_time' in mapped_df.columns and 'reassignment_count' in mapped_df.columns:
                        scaler = StandardScaler()
                        normalized_resolution_time = scaler.fit_transform(mapped_df[['resolution_time']])
                        normalized_reassignment_count = scaler.fit_transform(mapped_df[['reassignment_count']])

                        def calculate_complexity(res_time, reassign_count):
                            if pd.isna(res_time) or pd.isna(reassign_count):
                                return 'Unknown'
                            score = res_time + reassign_count
                            if score < -1: return 'Simple'
                            elif score < 0: return 'Medium'
                            elif score < 1: return 'Hard'
                            else: return 'Complex'

                        mapped_df['complexity'] = [
                            calculate_complexity(rt, rc) 
                            for rt, rc in zip(normalized_resolution_time, normalized_reassignment_count)
                        ]

                        # Customer Satisfaction Calculation
                        logger.info("Starting customer satisfaction score calculation")
                    try:
                        # Calculate all satisfaction scores
                        scored_df = calculate_customer_satisfaction(mapped_df)
                        
                        # Update mapped_df with all the scores
                        score_columns = [
                            'resolution_time_score', 
                            'reassignment_score', 
                            'reopen_score', 
                            'sentiment_score', 
                            'customer_satisfaction'
                        ]
                        
                        for col in score_columns:
                            mapped_df[col] = scored_df[col]
                            
                        logger.info("Successfully calculated customer satisfaction scores")
                    except Exception as e:
                        logger.error(f"Error in customer satisfaction calculation: {e}")
                        # Set default values if calculation fails
                        mapped_df['customer_satisfaction'] = 'neutral'
                        mapped_df['resolution_time_score'] = '3'
                        mapped_df['reassignment_score'] = '3'
                        mapped_df['reopen_score'] = '3'
                        mapped_df['sentiment_score'] = '3'

                # Database Saving
                if "database_saving" not in completed_steps:
                    completed_steps.add("database_saving")
                    yield json.dumps(progress_steps["database_saving"]) + "\n"
                    
                    # Save to database with all scores
                    safe_email = current_user.email.replace('@', '_').replace('.', '_')
                    engine = create_engine(f'sqlite:///./zinc_{safe_email}.db')
                    
                    mapped_df.to_sql(
                        name=f"incident_data_{safe_email}",
                        con=engine,
                        if_exists='replace',
                        index=False
                    )

                # Mark processing as complete
                processing_complete = True
                yield json.dumps({
                    "progress": 100,
                    "message": "Processing complete!"
                }) + "\n"

            except Exception as e:
                logger.error(f"Error in event stream: {str(e)}", exc_info=True)
                error_message = f"Processing error: {str(e)}"
                if error_message not in sent_messages:
                    sent_messages.add(error_message)
                    yield json.dumps({
                        "error": error_message,
                        "progress": 100,
                        "eta": "Error occurred"
                    }) + "\n"
                processing_complete = True

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# After fetching the data into a DataFrame
def get_timing_analysis(df):
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Records with closed_at: {df['closed_at'].notna().sum()}")
    
    # Convert datetime columns to pandas datetime
    df['opened_at'] = pd.to_datetime(df['opened_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])
    
    # Hours analysis
    hourly_opened = df['opened_at'].dt.hour.value_counts().sort_index()
    hourly_closed = df['closed_at'].dt.hour.value_counts().sort_index()
    
    # Days analysis
    daily_opened = df['opened_at'].dt.dayofweek.value_counts().sort_index()
    daily_closed = df['closed_at'].dt.dayofweek.value_counts().sort_index()
    
    # Months analysis
    monthly_opened = df['opened_at'].dt.month.value_counts().sort_index()
    monthly_closed = df['closed_at'].dt.month.value_counts().sort_index()
    
    # Helper function to create final data structure
    def create_time_data(opened_series, closed_series, labels):
        result = []
        for i, label in enumerate(labels):
            result.append({
                "time": label,
                "opened": int(opened_series.get(i, 0)),
                "closed": int(closed_series.get(i, 0))
            })
        return result
    
    # Create labels
    hours = [f"{i:02d}:00" for i in range(24)]
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Create final data structure
    timing_data = {
        "hourly": create_time_data(hourly_opened, hourly_closed, hours),
        "daily": create_time_data(daily_opened, daily_closed, days),
        "monthly": create_time_data(monthly_opened, monthly_closed, months)
    }
    
    # Log sample data for verification
    logger.info("\nSample timing data:")
    for category, data in timing_data.items():
        logger.info(f"\n{category.upper()} data (first 3 entries):")
        for entry in data[:3]:
            logger.info(f"Time: {entry['time']}, Opened: {entry['opened']}, Closed: {entry['closed']}")
    
    return timing_data

def safe_dict(obj):
    """Convert any non-serializable objects to strings"""
    if isinstance(obj, dict):
        return {k: safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_dict(x) for x in x]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj

def calculate_overall_score(df: pd.DataFrame) -> float:
    """Calculate overall performance score from 0-100"""
    try:
        logger.info(f"Starting overall score calculation with DataFrame shape: {df.shape}")
        
        # Log initial data types
        logger.info("Column dtypes:")
        for col in df.columns:
            logger.info(f"{col}: {df[col].dtype}")
        
        # Convert scores to numeric values
        score_columns = ['resolution_time_score', 'reassignment_score', 
                        'reopen_score', 'sentiment_score']
        
        # Log presence of required columns
        missing_columns = [col for col in score_columns + ['made_sla'] if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return 50.0

        # Log sample of score data before conversion
        logger.info("Sample of score data before conversion:")
        logger.info(df[score_columns].head().to_dict())
        
        numeric_scores = df[score_columns].apply(pd.to_numeric, errors='coerce')
        
        # Log any conversion issues
        null_counts = numeric_scores.isnull().sum()
        logger.info(f"Null counts after numeric conversion: {null_counts.to_dict()}")
        
        # Calculate base score (0-5 scale)
        weights = {
            'resolution_time_score': 0.3,
            'reassignment_score': 0.2,
            'reopen_score': 0.2,
            'sentiment_score': 0.3
        }
        
        # Log weighted calculation steps
        weighted_scores = sum(numeric_scores[col] * weight 
                            for col, weight in weights.items())
        logger.info(f"Weighted scores summary: mean={weighted_scores.mean()}, min={weighted_scores.min()}, max={weighted_scores.max()}")
        
        base_score = weighted_scores / sum(weights.values())
        logger.info(f"Base score summary: mean={base_score.mean()}, min={base_score.min()}, max={base_score.max()}")
        
        # Calculate SLA compliance percentage with explicit type checking
        logger.info("made_sla column info:")
        logger.info(f"made_sla dtype: {df['made_sla'].dtype}")
        logger.info(f"made_sla unique values: {df['made_sla'].unique()}")
        
        # Convert made_sla to numeric explicitly
        try:
            if df['made_sla'].dtype == bool:
                sla_numeric = df['made_sla'].astype(int)
            else:
                # Handle string 'True'/'False' or other cases
                sla_numeric = df['made_sla'].map({'True': 1, 'FALSE': 0, True: 1, False: 0})
            
            sla_compliance = sla_numeric.mean() * 100
            logger.info(f"SLA compliance calculated: {sla_compliance}")
            
        except Exception as sla_error:
            logger.error(f"Error in SLA calculation: {str(sla_error)}")
            logger.error(f"made_sla sample data: {df['made_sla'].head()}")
            sla_compliance = 50.0  # Default value
        
        # Calculate final score
        final_score = (((base_score.mean() - 1) / 4) * 100 * 0.7) + (sla_compliance * 0.3)
        logger.info(f"Final score before clamping: {final_score}")
        
        # Ensure score is between 0 and 100
        final_score = float(max(0, min(100, final_score)))
        logger.info(f"Final clamped score: {final_score}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating overall score: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return 50.0  # Return middle score on error

def generate_actionable_insights(df: pd.DataFrame) -> list:
    """Generate actionable insights from incident data."""
    insights = []
    
    try:
        logger.info("Starting actionable insights generation")
        logger.info(f"Input DataFrame shape: {df.shape}")
        
        # Log column names and data types
        logger.info("DataFrame columns and types:")
        for col in df.columns:
            logger.info(f"{col}: {df[col].dtype}")

        # 1. Resolution Time Analysis
        logger.info("Analyzing resolution times...")
        high_resolution_incidents = df[df['resolution_time'] > df['resolution_time'].mean() * 1.5]
        logger.info(f"Found {len(high_resolution_incidents)} high resolution time incidents")
        
        if not high_resolution_incidents.empty:
            common_category = high_resolution_incidents['category'].mode()[0]
            avg_time = high_resolution_incidents['resolution_time'].mean() / 60  # minutes
            logger.info(f"High resolution time category: {common_category}, avg time: {avg_time:.1f} minutes")
            
            insights.append({
                "category": "Resolution Time",
                "metric": "Average Resolution Time",
                "value": f"{avg_time:.1f} minutes",
                "trend": "up",
                "impact": "high",
                "recommendation": f"Focus on optimizing resolution time for {common_category} incidents. Consider creating standardized response procedures or additional training for this category."
            })

        # 2. Reassignment Pattern Analysis
        logger.info("Analyzing reassignment patterns...")
        high_reassign = df[df['reassignment_count'] > 2]
        if not high_reassign.empty:
            problem_subcategory = high_reassign['subcategory'].mode()[0]
            reassign_rate = (len(high_reassign) / len(df)) * 100
            logger.info(f"High reassignment rate: {reassign_rate:.1f}% in {problem_subcategory}")
            
            insights.append({
                "category": "Incident Routing",
                "metric": "High Reassignment Rate",
                "value": f"{reassign_rate:.1f}%",
                "trend": "up",
                "impact": "medium",
                "recommendation": f"High reassignment rate in {problem_subcategory}. Consider reviewing routing rules and team expertise mapping."
            })

        # 3. Customer Satisfaction Trends
        logger.info("Analyzing customer satisfaction...")
        negative_satisfaction = df[df['customer_satisfaction'].isin(['negative', 'highly negative'])]
        if not negative_satisfaction.empty:
            problem_symptom = negative_satisfaction['u_symptom'].mode()[0]
            negative_rate = (len(negative_satisfaction) / len(df)) * 100
            logger.info(f"Negative satisfaction rate: {negative_rate:.1f}% for {problem_symptom}")
            
            insights.append({
                "category": "Customer Satisfaction",
                "metric": "Negative Feedback Rate",
                "value": f"{negative_rate:.1f}%",
                "trend": "up",
                "impact": "high",
                "recommendation": f"High negative satisfaction related to '{problem_symptom}'. Review customer feedback and implement targeted improvements."
            })

        # 4. SLA Compliance
        logger.info("Analyzing SLA compliance...")
        # Convert made_sla to boolean (0 = False, 1 = True)
        sla_missed = df[df['made_sla'] == 0]
        if not sla_missed.empty:
            sla_category = sla_missed['category'].mode()[0]
            sla_miss_rate = (len(sla_missed) / len(df)) * 100
            logger.info(f"SLA miss rate: {sla_miss_rate:.1f}% in {sla_category}")
            
            insights.append({
                "category": "SLA Compliance",
                "metric": "SLA Miss Rate",
                "value": f"{sla_miss_rate:.1f}%",
                "trend": "up",
                "impact": "high",
                "recommendation": f"SLA breaches concentrated in {sla_category}. Review SLA thresholds and resource allocation."
            })

        # 5. Complexity Distribution
        logger.info("Analyzing complexity distribution...")
        complex_incidents = df[df['complexity'] == 'Complex']
        if not complex_incidents.empty:
            complex_category = complex_incidents['category'].mode()[0]
            complex_rate = (len(complex_incidents) / len(df)) * 100
            logger.info(f"Complex incidents rate: {complex_rate:.1f}% in {complex_category}")
            
            insights.append({
                "category": "Incident Complexity",
                "metric": "Complex Incidents Rate",
                "value": f"{complex_rate:.1f}%",
                "trend": "up",
                "impact": "medium",
                "recommendation": f"High concentration of complex incidents in {complex_category}. Consider specialized training and knowledge base improvements."
            })

        # 6. Workload Distribution Analysis
        logger.info("Analyzing workload distribution...")
        resolver_counts = df['resolved_by'].value_counts()
        avg_workload = resolver_counts.mean()
        high_workload_threshold = avg_workload * 1.5
        
        high_workload_resolvers = resolver_counts[resolver_counts > high_workload_threshold]
        
        if not high_workload_resolvers.empty:
            most_loaded_resolver = high_workload_resolvers.index[0]
            incident_count = high_workload_resolvers.iloc[0]
            workload_ratio = (incident_count / avg_workload)
            
            logger.info(f"Found {len(high_workload_resolvers)} resolvers with high workload")
            logger.info(f"Most loaded resolver: {most_loaded_resolver} with {incident_count} incidents")
            logger.info(f"Average workload: {avg_workload:.1f} incidents")
            
            # Get the most common category for this resolver
            resolver_incidents = df[df['resolved_by'] == most_loaded_resolver]
            common_category = resolver_incidents['category'].mode()[0]
            
            insights.append({
                "category": "Workload Distribution",
                "metric": "Resolver Workload Ratio",
                "value": f"{workload_ratio:.1f}x average",
                "trend": "up",
                "impact": "high" if workload_ratio > 2 else "medium",
                "recommendation": (
                    f"High incident concentration for resolver '{most_loaded_resolver}' "
                    f"({incident_count:.0f} incidents, {workload_ratio:.1f}x average load), "
                    f"particularly in '{common_category}'. Consider workload redistribution "
                    "or additional training for other team members in these areas."
                )
            })

        logger.info(f"Generated {len(insights)} actionable insights")
        logger.info("Sample insights:")
        for idx, insight in enumerate(insights):
            logger.info(f"Insight {idx + 1}:")
            logger.info(json.dumps(insight, indent=2))
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return []

@app.get("/report/{user_email}")
async def get_report(
    user_email: str, 
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Convert email to the same safe string format used in process_data
        safe_email = user_email.replace('@', '_').replace('.', '_')
        engine = create_engine(f'sqlite:///./zinc_{safe_email}.db')
        logger.info(f"Created engine for user {user_email}")
        
        with engine.connect() as connection:
            table_name = f"incident_data_{safe_email}"
            
            # Fetch all incident data into a DataFrame
            df = pd.read_sql(f"SELECT * FROM {table_name}", connection)

            # Generate insights with logging
            logger.info("Generating actionable insights...")
            insights = generate_actionable_insights(df)
            logger.info(f"Generated {len(insights)} insights")
            
            # Calculate card data
            # Most complex incidents - group by category, subcategory, and symptom
            most_complex = (
                df[df['complexity'] == 'Complex']
                .groupby(['category', 'subcategory', 'u_symptom'])
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False)
                .iloc[0] if len(df[df['complexity'] == 'Complex']) > 0 else pd.Series({
                    'category': 'N/A',
                    'subcategory': 'N/A',
                    'u_symptom': 'N/A',
                    'count': 0
                })
            ).to_dict()
            
            # Date range with proper parsing
            date_range = {
                'min_date': pd.to_datetime(df['opened_at']).min().strftime('%Y-%m-%d'),
                'max_date': pd.to_datetime(df['closed_at']).dropna().max().strftime('%Y-%m-%d')
            }
            
            # Average resolution time
            avg_resolution_time = {
                'avg_resolution_time': df['resolution_time'].mean() / 60  # Convert to minutes
            }
            
            # Get timing analysis
            timing_data = get_timing_analysis(df)
            
            # Create a clean DataFrame for table calculations
            table_df = df.copy()

            # Calculate table data with proper grouping and averaging
            table_data = (
                table_df.groupby(['category', 'subcategory', 'u_symptom'])
                .agg({
                    'number': 'count',  # Count of incidents in each group
                    'resolution_time': 'sum',  # Total resolution time for the group
                    'issue_description': lambda x: ' | '.join(x.unique()),  # Collect unique descriptions
                    'resolution_notes': lambda x: ' | '.join(x.unique())  # Collect unique resolution notes
                })
                .reset_index()
            )

            # Calculate average resolution time properly (total time / number of incidents)
            table_data['avg_resolution_time'] = (table_data['resolution_time'] / table_data['number']) / 60  # Convert to minutes

            # Rename columns for frontend
            table_data = table_data.rename(columns={
                'number': 'incident_count'
            }).to_dict('records')

            logger.info("Sample of table data calculations:")
            for row in table_data[:3]:
                logger.info(f"""
                Category: {row['category']}
                Subcategory: {row['subcategory']}
                Symptom: {row['u_symptom']}
                Count: {row['incident_count']}
                Avg Resolution Time (min): {row['avg_resolution_time']}
                Sample Description: {row['issue_description'][:100]}...
                Sample Resolution: {row['resolution_notes'][:100]}...
                """)

            # Calculate overall score
            overall_score = calculate_overall_score(df)
            
            # Get total incidents count
            total_incidents = len(df)

            # Get satisfaction counts for positive/negative groups
            satisfaction_counts = {
                'negative': len(df[df['customer_satisfaction'].isin(['negative', 'highly negative'])]),
                'positive': len(df[df['customer_satisfaction'].isin(['positive', 'highly positive'])])
            }

            # Get most negative incident group
            negative_group = df[df['customer_satisfaction'].isin(['negative', 'highly negative'])].groupby(
                ['category', 'subcategory', 'u_symptom']
            ).size().reset_index(name='count')
            most_negative = negative_group.nlargest(1, 'count').to_dict('records')[0] if not negative_group.empty else {
                'category': 'N/A',
                'subcategory': 'N/A',
                'u_symptom': 'N/A',
                'count': 0
            }

            # Get most positive incident group
            positive_group = df[df['customer_satisfaction'].isin(['positive', 'highly positive'])].groupby(
                ['category', 'subcategory', 'u_symptom']
            ).size().reset_index(name='count')
            most_positive = positive_group.nlargest(1, 'count').to_dict('records')[0] if not positive_group.empty else {
                'category': 'N/A',
                'subcategory': 'N/A',
                'u_symptom': 'N/A',
                'count': 0
            }

            # Get least complex incident group
            least_complex = df[df['complexity'] == 'Simple'].groupby(
                ['category', 'subcategory', 'u_symptom']
            ).size().reset_index(name='count')
            least_complex = least_complex.nlargest(1, 'count').to_dict('records')[0] if not least_complex.empty else {
                'category': 'N/A',
                'subcategory': 'N/A',
                'u_symptom': 'N/A',
                'count': 0
            }

            # Calculate sentiment distribution
            sentiment_distribution = (
                df['customer_satisfaction']
                .value_counts()
                .reindex(['highly positive', 'positive', 'neutral', 'negative', 'highly negative'])
                .dropna()  # Remove any sentiment categories that don't exist
                .to_dict()
            )

            report_data = {
                "cards": {
                    "total_incidents": total_incidents,
                    "most_complex": most_complex,
                    "least_complex": least_complex,
                    "most_negative": most_negative,
                    "most_positive": most_positive,
                    "date_range": safe_dict(date_range),
                    "avg_resolution_time": safe_dict(avg_resolution_time),
                    "overall_score": overall_score
                },
                "table_data": [safe_dict(row) for row in table_data],
                "timing_data": timing_data,
                "sentiment_distribution": sentiment_distribution,
                "actionable_insights": insights
            }
            
            # Log the final report data structure
            logger.info("Final report data structure:")
            logger.info(f"Cards: {len(report_data['cards'])} items")
            logger.info(f"Table data: {len(report_data['table_data'])} rows")
            logger.info(f"Timing data: {len(report_data['timing_data'])} items")
            logger.info(f"Sentiment distribution: {len(report_data['sentiment_distribution'])} items")
            logger.info(f"Actionable insights: {len(report_data['actionable_insights'])} items")
            
            return report_data
            
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/generate_sop")
async def generate_sop(data: dict):
    try:
        # Input validation logging
        logger.info("Starting SOP generation")
        logger.debug(f"Received data: {data}")
        
        issue_description = data.get("issue_description")
        resolution_notes = data.get("resolution_notes")
        
        # Validate inputs
        if not issue_description or not resolution_notes:
            logger.error("Missing required fields for SOP generation")
            raise HTTPException(
                status_code=400, 
                detail="Both issue_description and resolution_notes are required"
            )
            
        logger.info("Initializing Ollama LLM")
        try:
            llm = Ollama(model="phi3.5")
            logger.info("Successfully initialized Ollama LLM")
        except Exception as llm_error:
            logger.error(f"Failed to initialize Ollama LLM: {str(llm_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM initialization failed: {str(llm_error)}"
            )

        # Log template creation
        logger.info("Creating prompt template")
        template = """
        Given the following incident description and resolution notes, generate a detailed Standard Operating Procedure (SOP) to prevent similar incidents in the future and proactively address potential issues:

        Incident Description: {issue_description}

        Resolution Notes: {resolution_notes}

        Please provide a comprehensive SOP that includes:
        1. Steps to prevent this type of incident from occurring again
        2. Proactive measures to identify and address similar issues before they happen
        3. Best practices for handling this type of incident if it does occur
        4. Any necessary training or knowledge sharing to prevent future occurrences
        5. Order the steps based on what is most likely to have a greater impact on incident resolution for the users and agents 

        SOP:
        """

        try:
            prompt = PromptTemplate(
                template=template, 
                input_variables=["issue_description", "resolution_notes"]
            )
            logger.info("Successfully created prompt template")
        except Exception as prompt_error:
            logger.error(f"Failed to create prompt template: {str(prompt_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Prompt template creation failed: {str(prompt_error)}"
            )

        # Create and run chain
        logger.info("Creating LLM chain")
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            logger.info("Successfully created LLM chain")
            
            logger.info("Running LLM chain")
            # Use invoke instead of run
            result = await chain.ainvoke({
                "issue_description": issue_description,
                "resolution_notes": resolution_notes
            })
            
            logger.info("Successfully generated SOP")
            sop = result.get('text', '')  # Get the text from the result
            
            # Log a sample of the output (first 100 chars)
            logger.debug(f"Generated SOP preview: {sop[:100]}...")
            
            return {"sop": sop}
            
        except Exception as chain_error:
            logger.error(f"Failed during chain creation or execution: {str(chain_error)}")
            logger.error(f"Chain error type: {type(chain_error)}")
            logger.error(f"Chain error traceback", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"SOP generation failed: {str(chain_error)}"
            )

    except HTTPException as http_error:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in generate_sop: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during SOP generation: {str(e)}"
        )

# Add this near the top of the file with other helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class EngagementBase(BaseModel):
    name: str

class EngagementCreate(EngagementBase):
    pass

class EngagementResponse(EngagementBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        orm_mode = True

@app.get("/engagements", response_model=list[EngagementResponse])
async def get_engagements(
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    engagements = db.query(models.Engagement).filter(
        models.Engagement.user_id == current_user.email
    ).all()
    
    return engagements

def calculate_customer_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate customer satisfaction scores and return all component scores"""
    try:
        # Create a copy of the input dataframe to store all scores
        scored_df = df.copy()
        
        # Handle reassignment count (0-27 range)
        try:
            bins = [-float('inf'), 0, 1, 2, 4, float('inf')]
            labels = ['5', '4', '3', '2', '1']
            scored_df['reassignment_score'] = pd.cut(
                scored_df['reassignment_count'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
        except Exception as e:
            logger.warning(f"Error in reassignment binning: {e}")
            scored_df['reassignment_score'] = '3'
        
        # Handle reopen count (0-8 range)
        try:
            bins = [-float('inf'), 0, 1, 2, 3, float('inf')]
            labels = ['5', '4', '3', '2', '1']
            scored_df['reopen_score'] = pd.cut(
                scored_df['reopen_count'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
        except Exception as e:
            logger.warning(f"Error in reopen binning: {e}")
            scored_df['reopen_score'] = '3'
        
        # Handle resolution time with qcut
        try:
            scored_df['resolution_time_score'] = pd.qcut(
                scored_df['resolution_time'],
                q=5,
                labels=['5', '4', '3', '2', '1'],
                duplicates='drop'
            )
        except Exception as e:
            logger.warning(f"Error in resolution time binning: {e}")
            scored_df['resolution_time_score'] = '3'
            
        # Map sentiment scores
        sentiment_map = {
            'highly positive': '5',
            'positive': '4',
            'neutral': '3',
            'negative': '2',
            'highly negative': '1'
        }
        scored_df['sentiment_score'] = scored_df['sentiment'].map(sentiment_map).fillna('3')
        
        # Calculate weighted average
        try:
            weights = {
                'resolution_time_score': 0.3,
                'reassignment_score': 0.2,
                'reopen_score': 0.2,
                'sentiment_score': 0.3
            }
            
            # Convert scores to numeric for calculation
            numeric_scores = scored_df[list(weights.keys())].apply(pd.to_numeric, errors='coerce')
            
            # Calculate weighted average
            weighted_sum = sum(numeric_scores[col] * weight for col, weight in weights.items())
            total_weight = sum(weights.values())
            final_scores = weighted_sum / total_weight
            
            # Convert final scores to satisfaction levels
            scored_df['customer_satisfaction'] = pd.cut(
                final_scores,
                bins=[-float('inf'), 1.8, 2.6, 3.4, 4.2, float('inf')],
                labels=['highly negative', 'negative', 'neutral', 'positive', 'highly positive']
            ).fillna('neutral')
            
        except Exception as e:
            logger.error(f"Error in final score calculation: {e}")
            scored_df['customer_satisfaction'] = 'neutral'
        
        # Ensure all score columns are strings
        score_columns = ['resolution_time_score', 'reassignment_score', 'reopen_score', 
                        'sentiment_score', 'customer_satisfaction']
        for col in score_columns:
            scored_df[col] = scored_df[col].astype(str)
        
        return scored_df
        
    except Exception as e:
        logger.error(f"Error calculating satisfaction scores: {e}")
        # Return dataframe with default values
        df['customer_satisfaction'] = 'neutral'
        df['resolution_time_score'] = '3'
        df['reassignment_score'] = '3'
        df['reopen_score'] = '3'
        df['sentiment_score'] = '3'
        return df


