import os
import uuid
from pathlib import Path
from PIL import Image
from fastapi import UploadFile
import aiofiles

async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination and return the path"""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(destination, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return destination

def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename while preserving extension"""
    ext = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4().hex}{ext}"

def validate_image(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False

def get_image_dimensions(file_path: str) -> tuple:
    """Get image width and height"""
    img = Image.open(file_path)
    return img.size