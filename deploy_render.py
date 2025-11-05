#!/usr/bin/env python3
"""
Automated Render Deployment Script
Uses Render API to deploy the ML Platform programmatically.
"""
import requests
import json
import sys
import time
from typing import Dict, Any, Optional

# Render API Configuration
RENDER_API_BASE = "https://api.render.com/v1"
RENDER_API_KEY = "rnd_m4Ky2HffiJibZzOaNwnAsCaKJZFz"

class RenderDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.owner_id = None

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request to Render."""
        url = f"{RENDER_API_BASE}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PATCH":
                response = requests.patch(url, headers=self.headers, json=data)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            sys.exit(1)

    def get_owner(self) -> str:
        """Get the owner ID for the authenticated user."""
        print("🔍 Getting owner information...")
        response = self._make_request("GET", "/owners")

        if not response or len(response) == 0:
            print("❌ No owners found for this API key")
            sys.exit(1)

        owner = response[0]
        self.owner_id = owner['id']
        print(f"✅ Owner: {owner.get('name', owner['id'])}")
        return self.owner_id

    def create_database(self, name: str = "ml-platform-db") -> Dict:
        """Create PostgreSQL database."""
        print(f"\n📊 Creating PostgreSQL database: {name}")

        data = {
            "type": "postgres",
            "name": name,
            "databaseName": "reasoner_platform",
            "databaseUser": "platform_user",
            "plan": "free",
            "region": "oregon",
            "ownerId": self.owner_id
        }

        response = self._make_request("POST", "/postgres", data)
        db_id = response['id']

        print(f"✅ Database created: {db_id}")
        print(f"   Connection string will be available once provisioned")

        # Wait for database to provision
        print("⏳ Waiting for database to provision (this may take 2-3 minutes)...")
        time.sleep(10)

        return response

    def get_databases(self) -> list:
        """List all databases."""
        print("\n📋 Fetching existing databases...")
        response = self._make_request("GET", "/postgres")

        if isinstance(response, list):
            databases = response
        else:
            databases = response.get('postgres', [])

        for db in databases:
            print(f"   • {db['name']} ({db['id']})")

        return databases

    def create_backend_service(self, db_id: str, github_repo: str) -> Dict:
        """Create backend web service."""
        print(f"\n🚀 Creating backend service...")

        # Extract owner and repo from GitHub URL
        # Expected format: https://github.com/owner/repo or owner/repo
        if github_repo.startswith("https://github.com/"):
            github_repo = github_repo.replace("https://github.com/", "")

        parts = github_repo.strip("/").split("/")
        if len(parts) != 2:
            print(f"❌ Invalid GitHub repo format. Expected: owner/repo, got: {github_repo}")
            sys.exit(1)

        owner, repo = parts

        data = {
            "type": "web_service",
            "name": "ml-platform-backend",
            "ownerId": self.owner_id,
            "repo": f"https://github.com/{owner}/{repo}",
            "branch": "main",
            "rootDir": "backend",
            "region": "oregon",
            "plan": "free",
            "buildCommand": "pip install --upgrade pip && pip install -r requirements.txt",
            "startCommand": "chmod +x start.sh && ./start.sh",
            "healthCheckPath": "/health",
            "envVars": [
                {
                    "key": "DATABASE_URL",
                    "value": f"${{POSTGRES_{db_id}_URL}}"
                },
                {
                    "key": "SECRET_KEY",
                    "generateValue": True
                },
                {
                    "key": "ALGORITHM",
                    "value": "HS256"
                },
                {
                    "key": "ACCESS_TOKEN_EXPIRE_MINUTES",
                    "value": "30"
                },
                {
                    "key": "REFRESH_TOKEN_EXPIRE_DAYS",
                    "value": "7"
                },
                {
                    "key": "CORS_ORIGINS",
                    "value": "*"
                },
                {
                    "key": "APP_NAME",
                    "value": "Reasoner AI Platform"
                },
                {
                    "key": "APP_VERSION",
                    "value": "1.0.0"
                },
                {
                    "key": "API_V1_PREFIX",
                    "value": "/api/v1"
                },
                {
                    "key": "DEBUG",
                    "value": "False"
                },
                {
                    "key": "LOG_LEVEL",
                    "value": "INFO"
                },
                {
                    "key": "API_KEY_ENABLED",
                    "value": "false"
                },
                {
                    "key": "MLFLOW_TRACKING_URI",
                    "value": "sqlite:///./mlflow.db"
                }
            ]
        }

        response = self._make_request("POST", "/services", data)
        service_id = response['service']['id']

        print(f"✅ Backend service created: {service_id}")
        print(f"   URL: https://ml-platform-backend.onrender.com")

        return response

    def create_frontend_service(self, github_repo: str, backend_url: str) -> Dict:
        """Create frontend static site."""
        print(f"\n🎨 Creating frontend service...")

        # Extract owner and repo from GitHub URL
        if github_repo.startswith("https://github.com/"):
            github_repo = github_repo.replace("https://github.com/", "")

        parts = github_repo.strip("/").split("/")
        owner, repo = parts

        data = {
            "type": "static_site",
            "name": "ml-platform-frontend",
            "ownerId": self.owner_id,
            "repo": f"https://github.com/{owner}/{repo}",
            "branch": "main",
            "rootDir": "frontend",
            "region": "oregon",
            "buildCommand": "npm install && npm run build",
            "publishPath": "dist",
            "envVars": [
                {
                    "key": "VITE_API_URL",
                    "value": backend_url
                }
            ],
            "headers": [
                {
                    "path": "/*",
                    "name": "X-Frame-Options",
                    "value": "SAMEORIGIN"
                },
                {
                    "path": "/*",
                    "name": "X-Content-Type-Options",
                    "value": "nosniff"
                }
            ],
            "routes": [
                {
                    "type": "rewrite",
                    "source": "/*",
                    "destination": "/index.html"
                }
            ]
        }

        response = self._make_request("POST", "/services", data)
        service_id = response['service']['id']

        print(f"✅ Frontend service created: {service_id}")
        print(f"   URL: https://ml-platform-frontend.onrender.com")

        return response

    def list_services(self) -> list:
        """List all services."""
        print("\n📋 Fetching existing services...")
        response = self._make_request("GET", "/services")

        services = response if isinstance(response, list) else []

        for service in services:
            svc = service.get('service', service)
            print(f"   • {svc['name']} ({svc['type']}) - {svc.get('serviceDetails', {}).get('url', 'N/A')}")

        return services

    def deploy(self, github_repo: str):
        """Deploy the entire platform."""
        print("=" * 70)
        print("🚀 ML Platform - Automated Render Deployment")
        print("=" * 70)

        # Step 1: Get owner
        self.get_owner()

        # Step 2: Check existing resources
        existing_dbs = self.get_databases()
        existing_services = self.list_services()

        # Find existing database
        db = None
        for existing_db in existing_dbs:
            if existing_db['name'] == 'ml-platform-db':
                db = existing_db
                print(f"\n✅ Using existing database: {db['id']}")
                break

        # Step 3: Create database if not exists
        if not db:
            db = self.create_database()

        db_id = db['id']

        # Step 4: Create backend service
        backend = self.create_backend_service(db_id, github_repo)
        backend_url = f"https://{backend['service']['name']}.onrender.com"

        # Step 5: Create frontend service
        frontend = self.create_frontend_service(github_repo, backend_url)

        # Done!
        print("\n" + "=" * 70)
        print("✅ Deployment initiated successfully!")
        print("=" * 70)
        print(f"\n📍 Your Services:")
        print(f"   Backend:  {backend_url}")
        print(f"   Frontend: https://{frontend['service']['name']}.onrender.com")
        print(f"   API Docs: {backend_url}/docs")
        print(f"   Health:   {backend_url}/health")

        print(f"\n⏳ Services are now building...")
        print(f"   • Backend: Installing Python packages (~3-5 min)")
        print(f"   • Database: Provisioning PostgreSQL (~2-3 min)")
        print(f"   • Frontend: Building React app (~2-3 min)")

        print(f"\n📊 Monitor progress:")
        print(f"   Dashboard: https://dashboard.render.com")

        print(f"\n🔐 Default Admin Login:")
        print(f"   Email: admin@platform.local")
        print(f"   Password: admin123")
        print(f"   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")

        print(f"\n✨ Your ML Platform will be live in ~5-10 minutes!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy_render.py <github-repo>")
        print("Example: python deploy_render.py owner/repo")
        print("         python deploy_render.py https://github.com/owner/repo")
        sys.exit(1)

    github_repo = sys.argv[1]

    deployer = RenderDeployer(RENDER_API_KEY)
    deployer.deploy(github_repo)

if __name__ == "__main__":
    main()
