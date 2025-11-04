"""
Authentication API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS
)
from app.models.auth import User, RefreshToken
from app.schemas.auth_schemas import (
    UserCreate,
    UserLogin,
    AuthResponse,
    UserResponse,
    TokenResponse,
    RefreshTokenRequest
)

router = APIRouter()


@router.post("/register", response_model=AuthResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if user already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        role="user"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create tokens
    access_token = create_access_token(data={"sub": db_user.id})
    refresh_token_value = create_refresh_token()

    # Store refresh token
    refresh_token = RefreshToken(
        user_id=db_user.id,
        token=refresh_token_value,
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(refresh_token)
    db.commit()

    return AuthResponse(
        user=UserResponse(
            id=str(db_user.id),
            email=db_user.email,
            username=db_user.username,
            role=db_user.role,
            createdAt=db_user.created_at
        ),
        tokens=TokenResponse(
            accessToken=access_token,
            refreshToken=refresh_token_value
        )
    )


@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user."""
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    # Create tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token_value = create_refresh_token()

    # Store refresh token
    refresh_token = RefreshToken(
        user_id=user.id,
        token=refresh_token_value,
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(refresh_token)
    db.commit()

    return AuthResponse(
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            role=user.role,
            createdAt=user.created_at
        ),
        tokens=TokenResponse(
            accessToken=access_token,
            refreshToken=refresh_token_value
        )
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == request.refreshToken
    ).first()

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    if refresh_token.expires_at < datetime.utcnow():
        db.delete(refresh_token)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired"
        )

    # Create new access token
    access_token = create_access_token(data={"sub": refresh_token.user_id})

    return TokenResponse(
        accessToken=access_token,
        refreshToken=request.refreshToken
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        createdAt=current_user.created_at
    )


@router.post("/logout")
async def logout(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Logout user by invalidating refresh token."""
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == request.refreshToken
    ).first()

    if refresh_token:
        db.delete(refresh_token)
        db.commit()

    return {"message": "Successfully logged out"}
