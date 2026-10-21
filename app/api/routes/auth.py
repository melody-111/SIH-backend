import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from app.db.mongodb import MongoDBManager

logger = logging.getLogger("neerai.auth")
router = APIRouter(prefix="/auth", tags=["Authentication & Cloud Users"])

mongo = MongoDBManager()

class SignUpRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Full Name of the user")
    identifier: str = Field(..., min_length=3, max_length=120, description="Mobile number or email address")
    password: str = Field(..., min_length=4, max_length=128, description="Account password")
    boat_reg: Optional[str] = Field(None, description="Optional Boat registration number")

class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=3, max_length=120, description="Mobile number or email address")
    password: str = Field(..., min_length=1, max_length=128, description="Account password")

class GuestLoginRequest(BaseModel):
    nickname: Optional[str] = "Guest Navigator"

@router.post("/signup")
async def signup(req: SignUpRequest):
    """
    Registers a new user in MongoDB Atlas.
    Hashes password securely with bcrypt and returns access token.
    """
    clean_id = req.identifier.strip().lower()
    clean_name = req.name.strip()

    if not clean_id:
        raise HTTPException(status_code=400, detail="Mobile number or email address is required.")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Full name is required.")

    # Check for existing user
    existing = await mongo.find_user_by_identifier(clean_id)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="An account already exists with this mobile number or email. Please sign in instead."
        )

    try:
        user = await mongo.create_user(
            name=clean_name,
            identifier=clean_id,
            password=req.password,
            is_guest=False,
            boat_reg=req.boat_reg
        )
        token = mongo.create_jwt_token({
            "userId": user["userId"],
            "name": user["name"],
            "identifier": user["identifier"]
        })
        logger.info(f"New user registered successfully in MongoDB Atlas: {user['userId']} ({clean_id})")
        return {
            "success": True,
            "message": "Account created successfully in MongoDB Atlas.",
            "user": user,
            "token": token
        }
    except Exception as e:
        logger.error(f"Error registering user in MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Database error registering account: {str(e)}")

@router.post("/login")
async def login(req: LoginRequest):
    """
    Authenticates a user against MongoDB Atlas with password hash verification.
    """
    clean_id = req.identifier.strip().lower()

    user = await mongo.find_user_by_identifier(clean_id)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="No account found with this mobile number or email. Please check your credentials or create a new account."
        )

    hashed_pw = user.get("passwordHash")
    if not hashed_pw or not mongo.verify_password(req.password, hashed_pw):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password. Please try again."
        )

    await mongo.update_last_login(user["userId"])
    token = mongo.create_jwt_token({
        "userId": user["userId"],
        "name": user["name"],
        "identifier": user["identifier"]
    })

    user_clean = {
        "userId": user["userId"],
        "name": user["name"],
        "identifier": user["identifier"],
        "boatReg": user.get("boatReg"),
        "createdAt": user.get("createdAt"),
        "lastLoginAt": user.get("lastLoginAt"),
        "isGuest": user.get("isGuest", False),
        "preferences": user.get("preferences", {})
    }

    logger.info(f"User authenticated successfully via MongoDB Atlas: {user['userId']}")
    return {
        "success": True,
        "message": "Welcome back! Login successful.",
        "user": user_clean,
        "token": token
    }

@router.post("/guest")
async def guest_login(req: Optional[GuestLoginRequest] = None):
    """
    1-Tap fast guest session saved in MongoDB Atlas.
    """
    nickname = req.nickname if req and req.nickname else "Guest Navigator"
    guest_id = f"guest_{uuid.uuid4().hex[:8]}@neerai.local"

    try:
        user = await mongo.create_user(
            name=nickname,
            identifier=guest_id,
            password=None,
            is_guest=True
        )
        token = mongo.create_jwt_token({
            "userId": user["userId"],
            "name": user["name"],
            "identifier": user["identifier"],
            "isGuest": True
        })
        return {
            "success": True,
            "message": "Guest session initialized.",
            "user": user,
            "token": token
        }
    except Exception as e:
        logger.warning(f"Guest login MongoDB error (using offline fallback): {e}")
        now_iso = "2026-10-21T07:00:00Z"
        fallback_user = {
            "userId": f"guest_{uuid.uuid4().hex[:8]}",
            "name": nickname,
            "identifier": guest_id,
            "isGuest": True,
            "createdAt": now_iso
        }
        return {
            "success": True,
            "message": "Offline guest session initialized.",
            "user": fallback_user,
            "token": "guest_token_offline"
        }

@router.get("/me")
async def get_current_user_profile(authorization: Optional[str] = Header(None)):
    """
    Returns the currently authenticated user's profile from MongoDB Atlas.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication token required.")

    token = authorization.split(" ")[1]
    decoded = mongo.decode_jwt_token(token)
    if not decoded:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication session.")

    user_id = decoded.get("userId")
    user = await mongo.find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    user.pop("passwordHash", None)
    return {
        "success": True,
        "user": user
    }
