#!/bin/bash
set -e

echo "🚀 Starting ML Platform Backend..."

# Decode Google Drive credentials if provided
if [ ! -z "$GOOGLE_DRIVE_CREDENTIALS_BASE64" ]; then
    echo "🔑 Setting up Google Drive credentials..."
    echo "$GOOGLE_DRIVE_CREDENTIALS_BASE64" | base64 -d > /tmp/gd_credentials.json
    export GOOGLE_DRIVE_CREDENTIALS_PATH="/tmp/gd_credentials.json"
    echo "✅ Google Drive credentials ready"
fi

# Wait for database to be ready
echo "⏳ Waiting for database..."
python3 << END
import time
import sys
from sqlalchemy import create_engine, text
from app.core.config import settings

max_retries = 30
retry_interval = 2

for i in range(max_retries):
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database is ready!")
        sys.exit(0)
    except Exception as e:
        if i < max_retries - 1:
            print(f"⏳ Database not ready, retrying in {retry_interval}s... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
        else:
            print(f"❌ Database connection failed after {max_retries} attempts")
            sys.exit(1)
END

# Create database tables
echo "📊 Creating database tables..."
python3 << END
from app.core.database import engine
from app.models import database, auth, corrections, edge_devices, ethical_layer, safety_layer, chat, notifications

print("Creating all tables...")
database.Base.metadata.create_all(bind=engine)
print("✅ Database tables created successfully!")
END

# Create default admin user if not exists
echo "👤 Setting up default admin user..."
python3 << END
from app.core.database import SessionLocal
from app.models.auth import User, UserRole
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

db = SessionLocal()
try:
    # Check if admin exists
    admin = db.query(User).filter(User.email == "admin@platform.local").first()

    if not admin:
        # Create default admin
        admin = User(
            email="admin@platform.local",
            username="admin",
            hashed_password=pwd_context.hash("admin123"),
            role=UserRole.ADMIN,
            is_active=True
        )
        db.add(admin)
        db.commit()
        print("✅ Default admin user created!")
        print("   Email: admin@platform.local")
        print("   Password: admin123")
        print("   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")
    else:
        print("ℹ️  Admin user already exists")
finally:
    db.close()
END

echo "🎉 Setup complete! Starting server..."

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
