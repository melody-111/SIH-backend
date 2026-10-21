import logging
import datetime
import uuid
from typing import Optional, Dict, Any
import bcrypt
import jwt
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import MongoClient

from app.core.config import settings

logger = logging.getLogger("neerai.mongodb")

class MongoDBManager:
    """
    Asynchronous MongoDB Manager for NeerAI using Motor and PyMongo.
    Connects to MongoDB Atlas cluster for user authentication, profiles,
    and cloud data persistence.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._async_client = None
            cls._instance._sync_client = None
            cls._instance._db = None
            cls._instance._initialized = False
        return cls._instance

    @property
    def async_client(self) -> AsyncIOMotorClient:
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
        return self._async_client

    @property
    def sync_client(self) -> MongoClient:
        if self._sync_client is None:
            self._sync_client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
        return self._sync_client

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self.async_client[settings.MONGODB_DB_NAME]

    @property
    def users_collection(self) -> AsyncIOMotorCollection:
        return self.db["users"]

    @property
    def sessions_collection(self) -> AsyncIOMotorCollection:
        return self.db["sessions"]

    async def init_indexes(self):
        """Ensures unique indexes on identifier (email/phone) and userId."""
        try:
            users_col = self.users_collection
            await users_col.create_index("identifier", unique=True)
            await users_col.create_index("userId", unique=True)
            logger.info("MongoDB Atlas indexes verified successfully on 'users' collection.")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to ensure MongoDB indexes: {e}")

    # -------------------------------------------------------------------------
    # Password & Security Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def hash_password(password: str) -> str:
        """Hashes password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifies plain password against bcrypt hash."""
        if not plain_password or not hashed_password:
            return False
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @staticmethod
    def create_jwt_token(payload: Dict[str, Any], expires_days: int = 30) -> str:
        """Creates a signed JWT access token."""
        data = payload.copy()
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=expires_days)
        data.update({"exp": expire, "iat": datetime.datetime.now(datetime.timezone.utc)})
        return jwt.encode(data, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
        """Decodes and validates a JWT access token."""
        try:
            return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        except Exception as e:
            logger.warning(f"JWT decode error: {e}")
            return None

    # -------------------------------------------------------------------------
    # User CRUD Operations
    # -------------------------------------------------------------------------
    async def create_user(
        self,
        name: str,
        identifier: str,
        password: Optional[str] = None,
        is_guest: bool = False,
        boat_reg: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates and stores a new user record in MongoDB Atlas.
        """
        normalized_id = identifier.strip().lower()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        hashed_pw = self.hash_password(password) if password else None

        user_doc = {
            "userId": user_id,
            "name": name.strip(),
            "identifier": normalized_id,
            "boatReg": boat_reg.strip() if boat_reg else None,
            "passwordHash": hashed_pw,
            "isGuest": is_guest,
            "createdAt": now_iso,
            "updatedAt": now_iso,
            "lastLoginAt": now_iso,
            "preferences": {
                "language": "hi",
                "riskTolerance": "STANDARD",
                "defaultLocation": "Mumbai"
            }
        }

        col = self.users_collection
        res = await col.insert_one(user_doc)
        user_doc["_id"] = str(res.inserted_id)
        # Never return the password hash
        user_doc.pop("passwordHash", None)
        return user_doc

    async def find_user_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Finds user by email or mobile number."""
        normalized_id = identifier.strip().lower()
        col = self.users_collection
        user = await col.find_one({"identifier": normalized_id})
        if user:
            user["_id"] = str(user["_id"])
        return user

    async def find_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Finds user by userId."""
        col = self.users_collection
        user = await col.find_one({"userId": user_id})
        if user:
            user["_id"] = str(user["_id"])
        return user

    async def update_last_login(self, user_id: str):
        """Updates last login timestamp for user."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        col = self.users_collection
        await col.update_one({"userId": user_id}, {"$set": {"lastLoginAt": now_iso, "updatedAt": now_iso}})
