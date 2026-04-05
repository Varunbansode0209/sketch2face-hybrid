from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database import get_database
from app.models.user import UserCreate, User, UserInDB
from app.utils.security import get_password_hash, verify_password, decode_token
from datetime import datetime

security = HTTPBearer()

async def create_user(user: UserCreate, db) -> User:
    """Create a new user"""
    users_collection = db["users"]
    
    # Check if user exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user document
    user_dict = user.dict()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    user_dict["created_at"] = datetime.utcnow()
    
    result = await users_collection.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    
    return User(id=str(result.inserted_id), **user.dict(exclude={"password"}), created_at=user_dict["created_at"])

async def authenticate_user(email: str, password: str, db) -> Optional[UserInDB]:
    """Authenticate user with email and password"""
    users_collection = db["users"]
    user = await users_collection.find_one({"email": email})
    
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    user["_id"] = str(user["_id"])
    return UserInDB(**user)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db = Depends(get_database)
) -> User:
    """Get current authenticated user from token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    users_collection = db["users"]
    user = await users_collection.find_one({"email": email})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    user["_id"] = str(user["_id"])
    return User(id=user["_id"], email=user["email"], name=user["name"], role=user["role"], created_at=user["created_at"])

async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Verify current user is admin"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin access required."
        )
    return current_user

async def get_current_investigator_user(current_user: User = Depends(get_current_user)) -> User:
    """Verify current user is investigator or admin"""
    if current_user.role not in ["investigator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Investigator access required."
        )
    return current_user

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db = Depends(get_database)  # Will be None if MongoDB unavailable
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None (for testing)"""
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        if payload is None:
            return None
        
        email: str = payload.get("sub")
        if email is None:
            return None
        
        # Check if database is available
        if db is None:
            return None  # Can't verify user without database
        
        users_collection = db["users"]
        user = await users_collection.find_one({"email": email})
        
        if user is None:
            return None
        
        user["_id"] = str(user["_id"])
        return User(id=user["_id"], email=user["email"], name=user["name"], role=user["role"], created_at=user["created_at"])
    except:
        return None