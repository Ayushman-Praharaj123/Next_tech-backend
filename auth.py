from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from pydantic import BaseModel
from typing import Optional

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Security scheme
security = HTTPBearer()

# ARMY CREDENTIALS (CLASSIFIED)
ARMY_USERS = {
    "admin": {
        "username": os.getenv("ADMIN_USERNAME", "guard_admin"),
        "password": os.getenv("ADMIN_PASSWORD", "Army@Guard2024!"),
        "email": os.getenv("ADMIN_EMAIL", "admin@guardx.army.mil"),
        "full_name": os.getenv("ADMIN_FULL_NAME", "Guard-X Administrator"),
        "role": "ADMIN",
        "clearance_level": "TOP_SECRET",
        "unit": "CYBER_WARFARE_DIVISION"
    },
    "operator": {
        "username": os.getenv("OPERATOR_USERNAME", "field_operator"),
        "password": os.getenv("OPERATOR_PASSWORD", "Field@Ops2024!"),
        "email": os.getenv("OPERATOR_EMAIL", "operator@guardx.army.mil"),
        "full_name": os.getenv("OPERATOR_FULL_NAME", "Field Operator"),
        "role": "OPERATOR",
        "clearance_level": "SECRET",
        "unit": "SURVEILLANCE_OPERATIONS"
    }
}

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: dict
    clearance_level: str
    unit: str

def verify_password(plain_password, stored_password):
    """Verify password against stored password"""
    return plain_password == stored_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_army_user(username: str, password: str):
    """Authenticate against army credentials"""
    # Check admin credentials
    for user_type, user_data in ARMY_USERS.items():
        if user_data["username"] == username:
            if verify_password(password, user_data["password"]):
                return {
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "full_name": user_data["full_name"],
                    "role": user_data["role"],
                    "clearance_level": user_data["clearance_level"],
                    "unit": user_data["unit"],
                    "login_time": datetime.utcnow(),
                    "user_type": user_type
                }
    return False

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated army user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="UNAUTHORIZED ACCESS - INVALID MILITARY CREDENTIALS",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Verify user still exists in army system
    user_found = False
    for user_data in ARMY_USERS.values():
        if user_data["username"] == username:
            user_found = True
            return {
                "username": user_data["username"],
                "email": user_data["email"],
                "full_name": user_data["full_name"],
                "role": user_data["role"],
                "clearance_level": user_data["clearance_level"],
                "unit": user_data["unit"]
            }
    
    if not user_found:
        raise credentials_exception

def require_admin_access(current_user = Depends(get_current_user)):
    """Require admin level access"""
    if current_user.get("role") != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="INSUFFICIENT CLEARANCE - ADMIN ACCESS REQUIRED"
        )
    return current_user

def require_clearance_level(required_level: str):
    """Require specific clearance level"""
    def check_clearance(current_user = Depends(get_current_user)):
        user_clearance = current_user.get("clearance_level", "")
        clearance_hierarchy = ["PUBLIC", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
        
        if required_level not in clearance_hierarchy or user_clearance not in clearance_hierarchy:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="INVALID CLEARANCE LEVEL"
            )
        
        required_index = clearance_hierarchy.index(required_level)
        user_index = clearance_hierarchy.index(user_clearance)
        
        if user_index < required_index:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"INSUFFICIENT CLEARANCE - {required_level} LEVEL REQUIRED"
            )
        return current_user
    return check_clearance

# Initialize system on startup
def initialize_army_auth_system():
    """Initialize army authentication system"""
    print("🔒 INITIALIZING ARMY AUTHENTICATION SYSTEM")
    print("=" * 50)
    print("🎖️  CLASSIFIED MILITARY SYSTEM")
    print("🔐 AUTHORIZED PERSONNEL ONLY")
    print("=" * 50)
    
    # Log available users (without passwords)
    for user_type, user_data in ARMY_USERS.items():
        print(f"👤 {user_data['role']}: {user_data['username']}")
        print(f"   📧 {user_data['email']}")
        print(f"   🏛️  {user_data['unit']}")
        print(f"   🔒 Clearance: {user_data['clearance_level']}")
        print("-" * 30)
    
    print("✅ ARMY AUTH SYSTEM OPERATIONAL")

