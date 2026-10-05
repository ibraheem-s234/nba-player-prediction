
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Find the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load database credentials from the .env file
load_dotenv(PROJECT_ROOT / ".env")

# Build the database connection URL
database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME"),
)

# Create the SQLAlchemy engine
engine = create_engine(database_url)