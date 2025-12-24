# Standard library imports
from typing import Optional

# External package imports
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId

# Local application imports
from ...domain.repositories.user_repository import UserRepository
from ...domain.models.user import User
from ...domain.constants import UserFields
from .mongo_connection import get_user_collection


class MongoUserRepository(UserRepository):
    """MongoDB implementation of UserRepository"""
    
    def __init__(self, user_collection: Optional[AsyncIOMotorCollection] = None) -> None:
        self.user_collection = user_collection if user_collection is not None else get_user_collection()
    
    async def find_by_email(self, email: str) -> Optional[User]:
        """
        Find user by email address
        
        Args:
            email: Email address to search for
            
        Returns:
            User domain model if found, None otherwise
        """
        if not email:
            return None
        
        try:
            document = await self.user_collection.find_one({UserFields.EMAIL: email})
            if document is None:
                return None
            return self._document_to_user(document)
        except Exception as e:
            raise RuntimeError(f"Error finding user by email: {str(e)}")
    
    async def find_by_id(self, user_id: str) -> Optional[User]:
        """
        Find user by ID
        
        Args:
            user_id: User ID to search for
            
        Returns:
            User domain model if found, None otherwise
        """
        if not user_id:
            return None
        
        try:
            object_id = ObjectId(user_id)
        except (InvalidId, ValueError, TypeError):
            return None
        
        try:
            document = await self.user_collection.find_one({UserFields.MONGO_ID: object_id})
            if document is None:
                return None
            return self._document_to_user(document)
        except Exception as e:
            raise RuntimeError(f"Error finding user by ID: {str(e)}")
    
    async def save(self, user: User) -> User:
        """
        Save user (create new or update existing)
        
        Args:
            user: User domain model to save
            
        Returns:
            Saved User domain model with ID set
        """
        if not user:
            raise ValueError("User cannot be None")
        
        try:
            user_dict = self._user_to_dict(user)
            
            if user.id:
                # Update existing user
                try:
                    object_id = ObjectId(user.id)
                    update_result = await self.user_collection.update_one(
                        {UserFields.MONGO_ID: object_id},
                        {"$set": {k: v for k, v in user_dict.items() if k != UserFields.MONGO_ID}}
                    )
                    
                    if update_result.matched_count == 0:
                        raise ValueError(f"User with ID {user.id} not found")
                    
                    # Fetch and return updated document
                    updated_document = await self.user_collection.find_one({UserFields.MONGO_ID: object_id})
                    if updated_document is None:
                        raise RuntimeError(f"User {user.id} was updated but could not be retrieved")
                    
                    return self._document_to_user(updated_document)
                except (InvalidId, ValueError, TypeError):
                    raise ValueError(f"Invalid user ID format: {user.id}")
            else:
                # Create new user
                if UserFields.MONGO_ID in user_dict:
                    del user_dict[UserFields.MONGO_ID]
                
                result = await self.user_collection.insert_one(user_dict)
                
                # Fetch and return the newly created document
                new_document = await self.user_collection.find_one({UserFields.MONGO_ID: result.inserted_id})
                if new_document is None:
                    raise RuntimeError("User was created but could not be retrieved")
                
                return self._document_to_user(new_document)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error saving user: {str(e)}")
    
    def _document_to_user(self, document: dict) -> User:
        """
        Convert MongoDB document to User domain model
        
        Args:
            document: MongoDB document dictionary
            
        Returns:
            User domain model
        """
        if not document or UserFields.MONGO_ID not in document:
            raise ValueError("Invalid document: missing _id field")
        
        return User(
            id=str(document[UserFields.MONGO_ID]),
            full_name=document.get(UserFields.FULL_NAME, ""),
            email=document.get(UserFields.EMAIL, ""),
            hashed_password=document.get(UserFields.HASHED_PASSWORD, ""),
        )
    
    def _user_to_dict(self, user: User) -> dict:
        """
        Convert User domain model to MongoDB document
        
        Args:
            user: User domain model
            
        Returns:
            Dictionary ready for MongoDB storage
        """
        if not user:
            raise ValueError("User cannot be None")
        
        user_dict = {
            UserFields.FULL_NAME: user.full_name,
            UserFields.EMAIL: user.email,
            UserFields.HASHED_PASSWORD: user.hashed_password,
        }
        
        # Only include _id if user.id is valid
        if user.id:
            try:
                user_dict[UserFields.MONGO_ID] = ObjectId(user.id)
            except (InvalidId, ValueError, TypeError):
                # If ID is invalid, don't include it (will create new document)
                pass
        
        return user_dict

