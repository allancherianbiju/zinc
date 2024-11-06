from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Body, Response, Form
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
from werkzeug.utils import secure_filename
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

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

        # Convert frontend field names to internal names
        internal_mapping = {}
        for frontend_field, csv_column in mapping_data.mapping.items():
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
            else:
                logger.warning(f"Column {csv_field} not found in CSV file")
        logger.info("Mappings applied")

        # Convert datetime fields
        datetime_fields = ['opened_at', 'closed_at', 'resolved_at']
        required_fields = ['opened_at', 'closed_at']

        for field in datetime_fields:
            if field in mapped_df.columns:
                try:
                    # Parse dates and convert to Python datetime objects
                    mapped_df[field] = mapped_df[field].apply(parse_datetime)
                    
                    # Log some sample dates to verify parsing
                    sample_dates = mapped_df[field].dropna().head()
                    logger.info(f"Sample {field} dates after parsing:")
                    for date in sample_dates:
                        logger.info(f"  {date}")
                        
                except Exception as e:
                    if field in required_fields:
                        logger.error(f"Error parsing {field}: {str(e)}")
                        logger.error("Sample problematic values:")
                        problematic_values = mapped_df[field].head()
                        for val in problematic_values:
                            logger.error(f"  {val}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not parse {field} field. Please ensure dates are in DD-MM-YYYY format. Error: {str(e)}"
                        )
                    else:
                        logger.warning(f"Non-required datetime field {field} could not be parsed: {str(e)}")
            else:
                if field in required_fields:
                    frontend_field = next(k for k, v in FIELD_MAPPING.items() if v == field)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Required field '{frontend_field}' is missing from the mapping"
                    )

        # After parsing all dates, verify they're in a consistent format
        logger.info("Verifying datetime consistency...")
        for field in datetime_fields:
            if field in mapped_df.columns:
                invalid_dates = mapped_df[field].isna()
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum()} invalid dates in {field}")
                    logger.warning("Sample original values that couldn't be parsed:")
                    original_values = mapped_df.loc[invalid_dates, field].head()
                    for val in original_values:
                        logger.warning(f"  {val}")

        # Calculate resolution time using closed_at
        if 'opened_at' in mapped_df.columns and 'closed_at' in mapped_df.columns:
            resolution_time = (mapped_df['closed_at'] - mapped_df['opened_at']).dt.total_seconds()
            
            # Check for invalid resolution times
            invalid_times = resolution_time <= 0
            if invalid_times.any():
                problem_records = mapped_df[invalid_times][['number', 'opened_at', 'closed_at']]
                logger.error("Invalid resolution times found in the following records:")
                for _, record in problem_records.iterrows():
                    logger.error(
                        f"Ticket {record['number']}: "
                        f"Opened: {record['opened_at'].strftime('%d-%m-%Y %H:%M:%S')} -> "
                        f"Closed: {record['closed_at'].strftime('%d-%m-%Y %H:%M:%S')}"
                    )
                raise HTTPException(
                    status_code=400,
                    detail="Invalid resolution times detected: Some tickets have closing times before opening times"
                )
            
            mapped_df['resolution_time'] = resolution_time
            logger.info("Resolution time calculated using closed_at")
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot calculate resolution time: missing required timestamp fields"
            )

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

        # Create a unique table name for the user using email instead of id
        # Convert email to a safe string for table name
        safe_email = current_user.email.replace('@', '_').replace('.', '_')
        table_name = f"incident_data_{safe_email}"

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
        engine = create_engine(f'sqlite:///./zinc_{safe_email}.db')
        metadata.create_all(engine)

        # Insert data into the database
        with engine.connect() as connection:
            for _, row in mapped_df.iterrows():
                row_dict = row.to_dict()
                # Ensure all values are of the correct type for SQLite
                for key, value in row_dict.items():
                    if pd.isna(value):
                        if key == 'made_sla':
                            row_dict[key] = False
                        elif isinstance(value, (int, float)):
                            row_dict[key] = 0
                        else:
                            row_dict[key] = None
                    elif key in datetime_fields and value is not None:
                        # Ensure datetime fields are Python datetime objects
                        row_dict[key] = pd.to_datetime(value).to_pydatetime() if pd.notnull(value) else None
                try:
                    insert_stmt = incident_table.insert().values(**row_dict)
                    connection.execute(insert_stmt)
                except sqlalchemy.exc.IntegrityError as e:
                    continue
            connection.commit()

        logger.info(f"Data inserted into the database table: {table_name}")

        return {"message": "Data processed and inserted into the database successfully"}

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

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

            report_data = {
                "cards": {
                    "most_complex": safe_dict(most_complex),
                    "date_range": safe_dict(date_range),
                    "avg_resolution_time": safe_dict(avg_resolution_time)
                },
                "table_data": [safe_dict(row) for row in table_data],
                "timing_data": timing_data
            }
            
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
        5. Order the steps based on what is most likely to have a greater impact on incident resolution for the users and agents 

        SOP:
        """

        prompt = PromptTemplate(template=template, input_variables=["issue_description", "resolution_notes"])
        chain = LLMChain(llm=llm, prompt=prompt)

        sop = chain.run(issue_description=issue_description, resolution_notes=resolution_notes)

        return {"sop": sop}
    except Exception as e:
        logger.error(f"Error generating SOP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SOP: {str(e)}")

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



