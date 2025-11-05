"""
Load testing suite using Locust.

Run with: locust -f tests/load/locustfile.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between, tag
import json
import random
from datetime import datetime


class ReasonerUser(HttpUser):
    """Simulated user for load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Login and setup before tasks."""
        # Register or login
        self.username = f"loadtest_user_{random.randint(1000, 9999)}"
        self.email = f"{self.username}@example.com"

        # Register
        response = self.client.post("/api/v1/auth/register", json={
            "email": self.email,
            "username": self.username,
            "password": "TestPassword123!"
        }, name="/auth/register")

        if response.status_code in [200, 201]:
            data = response.json()
            self.access_token = data.get("accessToken")
            self.refresh_token = data.get("refreshToken")
        else:
            # Try login if user already exists
            response = self.client.post("/api/v1/auth/login", json={
                "username": self.username,
                "password": "TestPassword123!"
            }, name="/auth/login")
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("accessToken")
                self.refresh_token = data.get("refreshToken")

        # Set auth header
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

        # Create a test project
        response = self.client.post("/api/v1/projects",
            headers=self.headers,
            json={
                "name": f"Load Test Project {random.randint(1, 100)}",
                "description": "Project for load testing",
                "color": "#4A90E2"
            },
            name="/projects [create]"
        )
        if response.status_code in [200, 201]:
            self.project_id = response.json().get("id")
        else:
            self.project_id = None

    @task(5)
    @tag('read', 'health')
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="/health")

    @task(3)
    @tag('read', 'auth')
    def get_current_user(self):
        """Test current user endpoint."""
        self.client.get("/api/v1/auth/me", headers=self.headers, name="/auth/me")

    @task(10)
    @tag('read', 'projects')
    def list_projects(self):
        """Test listing projects."""
        self.client.get("/api/v1/projects", headers=self.headers, name="/projects [list]")

    @task(2)
    @tag('write', 'projects')
    def create_project(self):
        """Test creating a project."""
        self.client.post("/api/v1/projects",
            headers=self.headers,
            json={
                "name": f"Project {random.randint(1, 1000)}",
                "description": f"Test project created at {datetime.utcnow().isoformat()}",
                "color": random.choice(["#4A90E2", "#50C878", "#FF6B6B", "#FFA500"])
            },
            name="/projects [create]"
        )

    @task(7)
    @tag('read', 'conversations')
    def list_conversations(self):
        """Test listing conversations."""
        if self.project_id:
            self.client.get(f"/api/v1/projects/{self.project_id}/conversations",
                headers=self.headers,
                name="/conversations [list]"
            )

    @task(3)
    @tag('write', 'conversations')
    def create_conversation(self):
        """Test creating a conversation."""
        if self.project_id:
            response = self.client.post(f"/api/v1/projects/{self.project_id}/conversations",
                headers=self.headers,
                json={
                    "title": f"Conversation {random.randint(1, 1000)}",
                    "systemPrompt": "You are a helpful assistant."
                },
                name="/conversations [create]"
            )

    @task(4)
    @tag('write', 'messages')
    def send_message(self):
        """Test sending a message."""
        if self.project_id:
            # Get or create a conversation
            response = self.client.get(f"/api/v1/projects/{self.project_id}/conversations",
                headers=self.headers,
                name="/conversations [list for message]"
            )
            if response.status_code == 200:
                conversations = response.json()
                if conversations:
                    conversation_id = conversations[0]['id']
                    # Send message
                    self.client.post(f"/api/v1/conversations/{conversation_id}/messages",
                        headers=self.headers,
                        json={
                            "content": f"Test message at {datetime.utcnow().isoformat()}",
                            "role": "user"
                        },
                        name="/messages [create]"
                    )

    @task(6)
    @tag('read', 'notifications')
    def get_notifications(self):
        """Test getting notifications."""
        self.client.get("/api/v1/notifications",
            headers=self.headers,
            params={"limit": 20},
            name="/notifications [list]"
        )

    @task(4)
    @tag('read', 'notifications')
    def get_unread_count(self):
        """Test getting unread notification count."""
        self.client.get("/api/v1/notifications/unread-count",
            headers=self.headers,
            name="/notifications/unread-count"
        )

    @task(2)
    @tag('read', 'reports')
    def list_reports(self):
        """Test listing reports."""
        self.client.get("/api/v1/reports",
            headers=self.headers,
            params={"limit": 20},
            name="/reports [list]"
        )

    @task(1)
    @tag('read', 'admin')
    def get_system_metrics(self):
        """Test system metrics endpoint."""
        self.client.get("/api/v1/analytics/system", headers=self.headers, name="/analytics/system")

    @task(8)
    @tag('read', 'formulas')
    def list_formulas(self):
        """Test listing formulas."""
        self.client.get("/api/v1/formulas",
            params={"limit": 50},
            name="/formulas [list]"
        )


class AdminUser(HttpUser):
    """Simulated admin user for load testing admin endpoints."""

    wait_time = between(2, 5)

    def on_start(self):
        """Login as admin."""
        # Try to login with admin credentials
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        }, name="/auth/login [admin]")

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("accessToken")
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            self.headers = {}

    @task(5)
    @tag('admin', 'read')
    def get_all_users(self):
        """Test getting all users (admin only)."""
        self.client.get("/api/v1/admin/users", headers=self.headers, name="/admin/users")

    @task(3)
    @tag('admin', 'read')
    def get_system_metrics(self):
        """Test getting system metrics."""
        self.client.get("/api/v1/admin/metrics", headers=self.headers, name="/admin/metrics")


class StressTestUser(HttpUser):
    """User for stress testing with rapid requests."""

    wait_time = between(0.1, 0.5)  # Very short wait time

    def on_start(self):
        """Quick setup."""
        self.headers = {}

    @task(20)
    @tag('stress', 'health')
    def rapid_health_check(self):
        """Rapid health checks."""
        self.client.get("/health", name="/health [stress]")

    @task(10)
    @tag('stress', 'api')
    def rapid_api_calls(self):
        """Rapid API calls."""
        self.client.get("/api/v1/formulas", params={"limit": 10}, name="/formulas [stress]")
