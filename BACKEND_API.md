# Backend API Documentation

Complete API documentation for the ML Framework backend with authentication, chat, projects, and file upload support.

## Base URL
```
http://localhost:8000
```

## API Prefix
```
/api/v1
```

---

## Authentication Endpoints

### Register User
**POST** `/api/v1/auth/register`

Creates a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "user": {
    "id": "1",
    "email": "user@example.com",
    "username": "johndoe",
    "role": "user",
    "createdAt": "2025-01-01T00:00:00"
  },
  "tokens": {
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "secure-random-token"
  }
}
```

### Login
**POST** `/api/v1/auth/login`

Authenticate user and get tokens.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:** Same as register

### Refresh Token
**POST** `/api/v1/auth/refresh`

Refresh access token using refresh token.

**Request Body:**
```json
{
  "refreshToken": "secure-random-token"
}
```

**Response:**
```json
{
  "accessToken": "new-access-token",
  "refreshToken": "same-refresh-token"
}
```

### Get Current User
**GET** `/api/v1/auth/me`

Get current logged-in user information.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "id": "1",
  "email": "user@example.com",
  "username": "johndoe",
  "role": "user",
  "createdAt": "2025-01-01T00:00:00"
}
```

### Logout
**POST** `/api/v1/auth/logout`

Logout user by invalidating refresh token.

**Request Body:**
```json
{
  "refreshToken": "secure-random-token"
}
```

---

## Project Endpoints

### Create Project
**POST** `/api/v1/projects`

Create a new project.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Request Body:**
```json
{
  "name": "My Project",
  "description": "Project description (optional)",
  "color": "blue"
}
```

**Response:**
```json
{
  "id": "1",
  "name": "My Project",
  "description": "Project description",
  "color": "blue",
  "conversations": [],
  "createdAt": "2025-01-01T00:00:00",
  "updatedAt": "2025-01-01T00:00:00"
}
```

### Get All Projects
**GET** `/api/v1/projects`

Get all projects for the current user.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Response:**
```json
[
  {
    "id": "1",
    "name": "My Project",
    "description": "Project description",
    "color": "blue",
    "conversations": [...],
    "createdAt": "2025-01-01T00:00:00",
    "updatedAt": "2025-01-01T00:00:00"
  }
]
```

### Update Project
**PATCH** `/api/v1/projects/{project_id}`

Update a project.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Request Body:**
```json
{
  "name": "Updated Name",
  "description": "Updated description",
  "color": "green"
}
```

### Delete Project
**DELETE** `/api/v1/projects/{project_id}`

Delete a project and all its conversations.

**Headers:**
```
Authorization: Bearer {accessToken}
```

---

## Conversation Endpoints

### Create Conversation
**POST** `/api/v1/conversations`

Create a new conversation.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Request Body:**
```json
{
  "title": "New Conversation",
  "project_id": 1
}
```

### Get All Conversations
**GET** `/api/v1/conversations`

Get all conversations for the current user.

**Headers:**
```
Authorization: Bearer {accessToken}
```

### Get Conversation
**GET** `/api/v1/conversations/{conversation_id}`

Get a specific conversation with all messages.

**Headers:**
```
Authorization: Bearer {accessToken}
```

---

## Message Endpoints

### Send Message
**POST** `/api/v1/conversations/{conversation_id}/messages`

Send a message in a conversation.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Request Body:**
```json
{
  "content": "Hello, how can you help me?",
  "internet_enabled": true,
  "files": ["file_id_1", "file_id_2"]
}
```

**Response:**
```json
{
  "id": "1",
  "role": "assistant",
  "content": "I can help you with...",
  "timestamp": "2025-01-01T00:00:00",
  "attachments": []
}
```

---

## File Upload Endpoints

### Upload File
**POST** `/api/v1/upload`

Upload a file (supports all types: images, videos, audio, documents, CAD, XER, ZIP, etc.)

**Headers:**
```
Authorization: Bearer {accessToken}
Content-Type: multipart/form-data
```

**Request:**
- Form field: `file` (binary)

**Response:**
```json
{
  "filename": "document.pdf",
  "file_path": "/uploads/1/document.pdf",
  "file_size": 1024000,
  "mime_type": "application/pdf"
}
```

**Supported File Types:**
- **Images**: JPG, PNG, GIF, BMP, WebP
- **Videos**: MP4, MOV, AVI, MKV, WebM
- **Audio**: MP3, WAV, OGG, WebM
- **Documents**: PDF, DOC, DOCX, TXT, RTF
- **Archives**: ZIP, RAR, 7Z, TAR, GZ
- **Project Files**: XER, XML, MPP
- **CAD Files**: DWG, DXF, DWF, DGN, RVT, IFC, STEP, IGES, STL, 3DM
- **Code**: All text-based files
- **And more**: All file types supported

---

## Admin Endpoints

### Get All Users
**GET** `/api/v1/admin/users`

Get all users (admin only).

**Headers:**
```
Authorization: Bearer {accessToken}
```

### Update User Role
**PATCH** `/api/v1/admin/users/{user_id}`

Update user role (admin only).

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Request Body:**
```json
{
  "role": "admin"
}
```

### Delete User
**DELETE** `/api/v1/admin/users/{user_id}`

Delete a user (admin only).

**Headers:**
```
Authorization: Bearer {accessToken}
```

### Get System Metrics
**GET** `/api/v1/admin/metrics`

Get system-wide metrics (admin only).

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "totalUsers": 100,
  "totalProjects": 250,
  "totalConversations": 500,
  "totalMessages": 5000,
  "activeUsers24h": 50
}
```

---

## Error Responses

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden
```json
{
  "detail": "Not enough permissions"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 400 Bad Request
```json
{
  "detail": "Invalid input data"
}
```

---

## Authentication Flow

1. **Register/Login**: Get access token and refresh token
2. **Store Tokens**: Frontend stores tokens in localStorage
3. **API Requests**: Include access token in Authorization header
4. **Token Expiry**: Access token expires in 30 minutes
5. **Refresh**: Use refresh token to get new access token
6. **Logout**: Invalidate refresh token

---

## Database Schema

### Users Table
- id (Primary Key)
- email (Unique)
- username (Unique)
- hashed_password
- role (user/admin)
- is_active
- created_at
- updated_at

### Projects Table
- id (Primary Key)
- user_id (Foreign Key)
- name
- description
- color
- created_at
- updated_at

### Conversations Table
- id (Primary Key)
- project_id (Foreign Key, nullable)
- user_id (Foreign Key)
- title
- created_at
- updated_at

### Messages Table
- id (Primary Key)
- conversation_id (Foreign Key)
- role (user/assistant)
- content
- metadata (JSON)
- created_at

### FileAttachments Table
- id (Primary Key)
- message_id (Foreign Key)
- filename
- file_type
- file_size
- file_path
- mime_type
- created_at

---

## Running the Backend

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation
Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

---

## Environment Variables

Create a `.env` file:

```env
# Application
APP_NAME=ML Framework
APP_VERSION=1.0.0
DEBUG=True

# Database
DATABASE_URL=postgresql://user:password@localhost/mlframework

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

---

## Frontend Integration

The frontend is already configured to use these endpoints. The API service is located at:
```
frontend/src/services/api.ts
```

All requests automatically include:
- Authorization headers
- Token refresh on 401 errors
- Error handling
- Type safety with TypeScript

---

## Features Integrated

✅ JWT Authentication
✅ User Management
✅ Project Organization
✅ Conversation Management
✅ Real-time Messaging
✅ File Upload (All Types)
✅ Admin Panel
✅ System Metrics
✅ Role-Based Access Control
✅ Password Hashing
✅ Token Refresh
✅ CORS Support
✅ Database Models
✅ Type Safety (Pydantic)

---

## Next Steps

1. Set up PostgreSQL database
2. Configure environment variables
3. Run database migrations
4. Create first admin user
5. Start backend server
6. Start frontend development server
7. Test full integration

Enjoy your fully integrated ML Framework! 🚀
