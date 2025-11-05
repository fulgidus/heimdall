"""
FastAPI router for audio library management.

Endpoints:
- POST /upload - Upload audio file
- GET /list - List audio files with filters
- GET /{id} - Get audio metadata
- GET /{id}/download - Download audio file
- PATCH /{id}/enable - Enable/disable audio file
- PATCH /{id}/metadata - Update audio metadata
- DELETE /{id} - Delete audio file
- GET /random/sample - Get random audio sample for training
- GET /stats - Get library statistics
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import structlog

from ..models.audio_library import (
    AudioMetadata,
    AudioCategory,
    AudioUploadRequest,
    AudioUploadResponse,
    AudioListRequest,
    AudioListResponse,
    AudioUpdateRequest,
    AudioLibraryEmptyError,
    AudioLibraryStats,
    AudioFormat,
    CategoryWeights,
)
from ..storage.audio_storage import AudioLibraryStorage, get_audio_storage

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/audio-library", tags=["audio-library"])


@router.post("/upload", response_model=AudioUploadResponse, status_code=201)
async def upload_audio_file(
    file: UploadFile = File(..., description="Audio file (WAV/MP3/FLAC/OGG)"),
    category: AudioCategory = Form(..., description="Audio category"),
    tags: str = Form(default="", description="Comma-separated tags"),
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Upload audio file to library.
    
    Supports: WAV, MP3, FLAC, OGG formats
    
    Automatically extracts metadata (duration, sample rate, channels)
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        audio_format = AudioFormat.from_filename(file.filename)
        if not audio_format:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {AudioFormat.supported_extensions()}"
            )
        
        # Parse tags
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Upload
        file_data = await file.read()
        metadata = await storage.upload_audio(
            file_data=BytesIO(file_data),
            filename=file.filename,
            category=category,
            tags=tags_list
        )
        
        logger.info(
            "Audio file uploaded via API",
            audio_id=metadata.id,
            filename=file.filename,
            category=category.value,
            size_mb=metadata.file_size_bytes / 1024 / 1024
        )
        
        return AudioUploadResponse(
            metadata=metadata,
            message="Audio file uploaded successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload audio file", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list", response_model=AudioListResponse)
async def list_audio_files(
    category: Optional[AudioCategory] = Query(None, description="Filter by category"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status (true=enabled, false=disabled, null=all)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags (AND logic)"),
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    List audio files with optional filtering.
    
    Filters:
    - category: Filter by specific category
    - enabled: Filter by enabled status (true=enabled only, false=disabled only, null=all files)
    - tags: Filter by tags with AND logic (comma-separated)
    """
    try:
        # Parse tags
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
        
        # List files
        files = await storage.list_audio_files(
            category=category,
            enabled=enabled,
            tags=tags_list
        )
        
        return AudioListResponse(
            total=len(files),
            files=files
        )
    
    except Exception as e:
        logger.error("Failed to list audio files", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/weights", response_model=CategoryWeights)
async def get_category_weights(
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Get current category weights for training sample selection.
    
    Returns proportional weights (0.0-1.0) for each category.
    Used by training service to control sampling probabilities.
    """
    try:
        weights = await storage.get_category_weights()
        return weights
    
    except Exception as e:
        logger.error("Failed to get category weights", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get weights: {str(e)}")


@router.put("/weights", response_model=CategoryWeights)
async def update_category_weights(
    weights: CategoryWeights,
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Update category weights for training sample selection.
    
    Weights are proportional values (0.0-1.0) that control sampling probability.
    Backend automatically normalizes to ensure they sum to 1.0.
    
    Example: {voice: 0.4, music: 0.3, documentary: 0.2, conference: 0.1, custom: 0.0}
    """
    try:
        # Normalize weights to probability distribution
        normalized_weights = weights.normalize()
        
        # Save to storage
        await storage.set_category_weights(normalized_weights)
        
        logger.info(
            "Category weights updated",
            voice=normalized_weights.voice,
            music=normalized_weights.music,
            documentary=normalized_weights.documentary,
            conference=normalized_weights.conference,
            custom=normalized_weights.custom
        )
        
        return normalized_weights
    
    except Exception as e:
        logger.error("Failed to update category weights", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")


@router.get("/random/sample", response_model=AudioMetadata)
async def get_random_audio_sample(
    category: Optional[AudioCategory] = Query(None, description="Filter by category"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Get random enabled audio file for training.
    
    Returns random audio sample from enabled files.
    Used by training service to load realistic audio samples.
    """
    try:
        metadata = await storage.get_random_audio_sample(category=category, seed=seed)
        return metadata
    
    except AudioLibraryEmptyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"No enabled audio files available. {str(e)}"
        )
    except Exception as e:
        logger.error("Failed to get random audio sample", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get sample: {str(e)}")


@router.get("/stats", response_model=AudioLibraryStats)
async def get_library_statistics(
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Get audio library statistics.
    
    Returns:
    - File counts (total, enabled, disabled)
    - Total duration and storage size
    - Counts by category and format
    """
    try:
        stats = await storage.get_library_stats()
        return stats
    
    except Exception as e:
        logger.error("Failed to get library stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{audio_id}", response_model=AudioMetadata)
async def get_audio_metadata(
    audio_id: str,
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """Get metadata for specific audio file."""
    try:
        metadata = await storage.get_audio_metadata(audio_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_id}")
        
        return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get audio metadata", audio_id=audio_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@router.get("/{audio_id}/download")
async def download_audio_file(
    audio_id: str,
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """Download audio file."""
    try:
        # Get metadata for filename and content type
        metadata = await storage.get_audio_metadata(audio_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_id}")
        
        # Download file
        audio_bytes = await storage.download_audio(audio_id)
        
        return StreamingResponse(
            BytesIO(audio_bytes),
            media_type=f"audio/{metadata.format.value}",
            headers={
                "Content-Disposition": f"attachment; filename={metadata.filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download audio file", audio_id=audio_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.patch("/{audio_id}/enable", response_model=AudioMetadata)
async def toggle_audio_enabled(
    audio_id: str,
    enabled: bool = Query(..., description="Enable (true) or disable (false)"),
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Enable or disable audio file (soft delete).
    
    Disabled files won't be used for training but remain in storage.
    """
    try:
        metadata = await storage.set_audio_enabled(audio_id, enabled)
        
        logger.info(
            "Audio enabled status toggled via API",
            audio_id=audio_id,
            enabled=enabled
        )
        
        return metadata
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to toggle audio enabled", audio_id=audio_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to toggle: {str(e)}")


@router.patch("/{audio_id}/metadata", response_model=AudioMetadata)
async def update_audio_metadata_endpoint(
    audio_id: str,
    update: AudioUpdateRequest,
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Update audio file metadata.
    
    Can update:
    - filename (must keep same extension)
    - category
    - tags (replaces existing)
    """
    try:
        metadata = await storage.update_audio_metadata(
            audio_id=audio_id,
            filename=update.filename,
            category=update.category,
            tags=update.tags
        )
        
        logger.info("Audio metadata updated via API", audio_id=audio_id)
        return metadata
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to update audio metadata", audio_id=audio_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.delete("/{audio_id}", status_code=204)
async def delete_audio_file(
    audio_id: str,
    storage: AudioLibraryStorage = Depends(get_audio_storage)
):
    """
    Hard delete audio file.
    
    Removes file from MinIO storage and metadata catalog.
    Use PATCH /{audio_id}/enable for soft delete instead.
    """
    try:
        await storage.delete_audio(audio_id)
        
        logger.info("Audio file deleted via API", audio_id=audio_id)
        return None
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to delete audio file", audio_id=audio_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
