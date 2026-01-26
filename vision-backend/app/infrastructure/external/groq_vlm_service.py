"""Groq Vision Language Model (VLM) service for image analysis."""
import base64
import json
import logging
from io import BytesIO
from typing import Dict, Any, Optional
import httpx
import numpy as np
from PIL import Image

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class GroqVLMService:
    """
    Generic service for interacting with Groq's Vision Language Model API.
    
    This service handles:
    - Converting images to base64 format
    - Calling Groq's chat completions API with vision support
    - Parsing JSON responses from VLM
    
    Can be used by any scenario that needs image analysis with custom prompts.
    """
    
    GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Default model for weapon detection
    DEFAULT_VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Timeout for API calls (seconds)
    API_TIMEOUT = 30.0
    
    def __init__(self):
        """Initialize Groq VLM Service with API key from settings."""
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = getattr(settings, 'vlm_model', self.DEFAULT_VLM_MODEL)
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy array (BGR image) to base64-encoded PNG string.
        
        Args:
            image: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            Base64-encoded image string (data URL format)
        """
        try:
            # Convert BGR to RGB
            rgb_image = image[:, :, ::-1]  # Reverse channels
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image.astype(np.uint8))
            
            # Save to bytes buffer
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def _crop_person_region(
        self,
        frame: np.ndarray,
        box: list[float],
        padding: int = 20
    ) -> np.ndarray:
        """
        Crop person region from frame using bounding box.
        
        Args:
            frame: Full frame image (H, W, 3)
            box: Bounding box [x1, y1, x2, y2]
            padding: Padding pixels around the box
            
        Returns:
            Cropped image region
        """
        h, w = frame.shape[:2]
        
        # Extract coordinates
        x1, y1, x2, y2 = box
        
        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        # Crop region
        cropped = frame[y1:y2, x1:x2]
        
        # Ensure minimum size
        if cropped.size == 0:
            return frame  # Return full frame if crop is invalid
        
        return cropped
    
    def analyze_image(
        self,
        image: np.ndarray,
        prompt: str,
        crop_box: Optional[list[float]] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generic method to analyze an image with a custom prompt using Groq VLM.
        
        Args:
            image: Image as numpy array (BGR format) or list of images
            prompt: Custom prompt for the VLM (should request JSON response)
            crop_box: Optional bounding box [x1, y1, x2, y2] to crop image region
            temperature: Sampling temperature (0.0-2.0), lower = more deterministic
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with:
            {
                "content": str,  # Raw text response from VLM
                "parsed_json": dict | None,  # Parsed JSON if response is valid JSON
                "raw_response": dict  # Full API response
            }
        """
        # Support both single image and list of images
        images = image if isinstance(image, list) else [image]
        return self.analyze_images(images, prompt, crop_box, temperature, max_tokens)
    
    def analyze_images(
        self,
        images: list[np.ndarray],
        prompt: str,
        crop_box: Optional[list[float]] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generic method to analyze multiple images with a custom prompt using Groq VLM.
        
        Args:
            images: List of images as numpy arrays (BGR format)
            prompt: Custom prompt for the VLM (should request JSON response)
            crop_box: Optional bounding box [x1, y1, x2, y2] to crop image region
            temperature: Sampling temperature (0.0-2.0), lower = more deterministic
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with:
            {
                "content": str,  # Raw text response from VLM
                "parsed_json": dict | None,  # Parsed JSON if response is valid JSON
                "raw_response": dict  # Full API response
            }
        """
        if not self.api_key:
            logger.error("GROQ_API_KEY not configured")
            return {
                "content": "",
                "parsed_json": None,
                "raw_response": {},
                "error": "VLM API key not configured"
            }
        
        try:
            # Crop image regions if box provided
            processed_images = []
            for img in images:
                if crop_box:
                    processed_img = self._crop_person_region(img, crop_box)
                else:
                    processed_img = img
                processed_images.append(processed_img)
            
            # Convert all images to base64
            content_items = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            # Add all images to content
            for img in processed_images:
                base64_image = self._numpy_to_base64(img)
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })
            
            # Prepare messages for chat API
            messages = [
                {
                    "role": "user",
                    "content": content_items
                }
            ]
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.debug(f"Calling Groq VLM API with model: {self.model}")
            
            with httpx.Client(timeout=self.API_TIMEOUT) as client:
                response = client.post(
                    self.GROQ_CHAT_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract content from response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    logger.warning("Empty response from Groq VLM API")
                    return {
                        "content": "",
                        "parsed_json": None,
                        "raw_response": result,
                        "error": "Empty response from VLM"
                    }
                
                # Try to parse JSON from response
                parsed_json = None
                try:
                    # Try to extract JSON from markdown code blocks if present
                    json_content = content
                    if "```json" in json_content:
                        json_start = json_content.find("```json") + 7
                        json_end = json_content.find("```", json_start)
                        json_content = json_content[json_start:json_end].strip()
                    elif "```" in json_content:
                        json_start = json_content.find("```") + 3
                        json_end = json_content.find("```", json_start)
                        json_content = json_content[json_start:json_end].strip()
                    
                    parsed_json = json.loads(json_content)
                except json.JSONDecodeError:
                    # Not JSON, that's okay - return raw content
                    parsed_json = None
                
                return {
                    "content": content,
                    "parsed_json": parsed_json,
                    "raw_response": result
                }
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Groq VLM API: {e.response.status_code} - {e.response.text}")
            return {
                "content": "",
                "parsed_json": None,
                "raw_response": {},
                "error": f"API error: {e.response.status_code}"
            }
        except httpx.TimeoutException:
            logger.error("Timeout while calling Groq VLM API")
            return {
                "content": "",
                "parsed_json": None,
                "raw_response": {},
                "error": "VLM API timeout"
            }
        except Exception as e:
            logger.error(f"Unexpected error in Groq VLM: {e}", exc_info=True)
            return {
                "content": "",
                "parsed_json": None,
                "raw_response": {},
                "error": f"Error: {str(e)}"
            }
