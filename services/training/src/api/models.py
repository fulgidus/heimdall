"""
REST API endpoints for ML model architecture registry.

Provides endpoints for:
- Listing all available architectures
- Getting details for a specific architecture
- Comparing multiple architectures
- Filtering by data type or architecture family
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from ..models.model_registry import (
    MODEL_REGISTRY,
    ModelArchitectureInfo,
    PerformanceMetrics,
    get_model_info,
    list_models,
    compare_models,
    get_recommended_model,
    get_models_by_badge,
    model_info_to_dict,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


# ============================================================================
# Response Models
# ============================================================================

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model."""
    
    expected_error_min_m: float
    expected_error_max_m: float
    accuracy_stars: int
    inference_time_min_ms: float
    inference_time_max_ms: float
    speed_stars: int
    parameters_millions: float
    vram_training_gb: float
    vram_inference_gb: float
    efficiency_stars: int
    speed_emoji: str
    memory_emoji: str
    accuracy_emoji: str


class ModelArchitectureResponse(BaseModel):
    """Model architecture response model."""
    
    id: str
    display_name: str
    description: str
    long_description: str
    data_type: str
    architecture_type: str
    architecture_emoji: str
    performance: PerformanceMetricsResponse
    badges: List[str]
    best_for: List[str]
    not_recommended_for: List[str]
    backbone: Optional[str] = None
    pretrained_weights: Optional[str] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    training_difficulty: str
    convergence_epochs: int
    recommended_batch_size: int
    paper_url: Optional[str] = None
    implementation_file: str


class ModelListResponse(BaseModel):
    """List of model architectures."""
    
    total: int = Field(..., description="Total number of models")
    architectures: List[ModelArchitectureResponse]


class ModelComparisonResponse(BaseModel):
    """Comparison of multiple models."""
    
    models: List[ModelArchitectureResponse]
    comparison_table: dict = Field(
        ..., 
        description="Side-by-side comparison of key metrics"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_to_response(info: ModelArchitectureInfo) -> ModelArchitectureResponse:
    """Convert ModelArchitectureInfo to response model."""
    return ModelArchitectureResponse(
        id=info.id,
        display_name=info.display_name,
        description=info.description,
        long_description=info.long_description,
        data_type=info.data_type,
        architecture_type=info.architecture_type,
        architecture_emoji=info.architecture_emoji,
        performance=PerformanceMetricsResponse(
            expected_error_min_m=info.performance.expected_error_min_m,
            expected_error_max_m=info.performance.expected_error_max_m,
            accuracy_stars=info.performance.accuracy_stars,
            inference_time_min_ms=info.performance.inference_time_min_ms,
            inference_time_max_ms=info.performance.inference_time_max_ms,
            speed_stars=info.performance.speed_stars,
            parameters_millions=info.performance.parameters_millions,
            vram_training_gb=info.performance.vram_training_gb,
            vram_inference_gb=info.performance.vram_inference_gb,
            efficiency_stars=info.performance.efficiency_stars,
            speed_emoji=info.performance.speed_emoji,
            memory_emoji=info.performance.memory_emoji,
            accuracy_emoji=info.performance.accuracy_emoji,
        ),
        badges=info.badges,
        best_for=info.best_for,
        not_recommended_for=info.not_recommended_for,
        backbone=info.backbone,
        pretrained_weights=info.pretrained_weights,
        input_shape=info.input_shape,
        output_shape=info.output_shape,
        training_difficulty=info.training_difficulty,
        convergence_epochs=info.convergence_epochs,
        recommended_batch_size=info.recommended_batch_size,
        paper_url=info.paper_url,
        implementation_file=info.implementation_file,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get(
    "/architectures",
    response_model=ModelListResponse,
    summary="List all available model architectures",
    description="Get a complete list of all 11 registered ML architectures with metadata.",
)
async def list_architectures(
    data_type: Optional[str] = Query(
        None,
        description="Filter by data type: spectrogram, iq_raw, features, hybrid",
    ),
    architecture_type: Optional[str] = Query(
        None,
        description="Filter by architecture: cnn, transformer, tcn, hybrid, mlp",
    ),
    badges: Optional[str] = Query(
        None,
        description="Filter by badges (comma-separated): RECOMMENDED,FASTEST,etc.",
    ),
) -> ModelListResponse:
    """
    List all available model architectures.
    
    Supports filtering by:
    - data_type: spectrogram, iq_raw, features, hybrid
    - architecture_type: cnn, transformer, tcn, hybrid, mlp
    - badges: RECOMMENDED, MAXIMUM_ACCURACY, FASTEST, etc.
    
    Returns:
        ModelListResponse with filtered list of models
    """
    try:
        models = list(MODEL_REGISTRY.values())
        
        # Filter by data_type
        if data_type:
            models = list_models(data_type=data_type)
        
        # Filter by architecture_type
        if architecture_type:
            models = list_models(architecture_type=architecture_type)
        
        # Filter by badges
        if badges:
            badge_list = [b.strip().upper() for b in badges.split(",")]
            models = [m for m in models if any(b in m.badges for b in badge_list)]
        
        logger.info(
            "architectures_listed",
            total=len(models),
            data_type_filter=data_type,
            architecture_filter=architecture_type,
            badge_filter=badges,
        )
        
        return ModelListResponse(
            total=len(models),
            architectures=[_convert_to_response(m) for m in models],
        )
    
    except Exception as e:
        logger.error("architectures_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list architectures: {str(e)}",
        )


@router.get(
    "/architectures/{model_id}",
    response_model=ModelArchitectureResponse,
    summary="Get details for a specific model architecture",
    description="Get complete metadata for a specific model by ID.",
)
async def get_architecture(model_id: str) -> ModelArchitectureResponse:
    """
    Get details for a specific model architecture.
    
    Args:
        model_id: Model identifier (e.g., "iq_transformer", "localization_net_convnext_large")
    
    Returns:
        ModelArchitectureResponse with complete metadata
    
    Raises:
        404: Model not found
    """
    try:
        model_info = get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model architecture '{model_id}' not found. "
                       f"Available: {list(MODEL_REGISTRY.keys())}",
            )
        
        logger.info("architecture_retrieved", model_id=model_id)
        
        return _convert_to_response(model_info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("architecture_retrieval_failed", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve architecture: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare multiple model architectures",
    description="Side-by-side comparison of 2-6 model architectures.",
)
async def compare_architectures(
    model_ids: List[str] = Query(
        ...,
        min_items=2,
        max_items=6,
        description="List of 2-6 model IDs to compare",
    ),
) -> ModelComparisonResponse:
    """
    Compare multiple model architectures side-by-side.
    
    Args:
        model_ids: List of 2-6 model IDs to compare
    
    Returns:
        ModelComparisonResponse with models and comparison table
    
    Raises:
        400: Invalid number of models or model not found
    """
    try:
        # Validate model IDs
        models = []
        for model_id in model_ids:
            model_info = get_model_info(model_id)
            if not model_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found. "
                           f"Available: {list(MODEL_REGISTRY.keys())}",
                )
            models.append(model_info)
        
        # Build comparison table
        comparison_table = {
            "accuracy": {
                m.id: f"±{m.performance.expected_error_min_m}-{m.performance.expected_error_max_m}m "
                      f"({'⭐' * m.performance.accuracy_stars})"
                for m in models
            },
            "speed": {
                m.id: f"{m.performance.inference_time_min_ms}-{m.performance.inference_time_max_ms}ms "
                      f"({m.performance.speed_emoji})"
                for m in models
            },
            "memory": {
                m.id: f"{m.performance.parameters_millions}M params, "
                      f"{m.performance.vram_training_gb}GB training "
                      f"({m.performance.memory_emoji})"
                for m in models
            },
            "badges": {m.id: ", ".join(m.badges) for m in models},
            "data_type": {m.id: m.data_type for m in models},
            "difficulty": {m.id: m.training_difficulty for m in models},
        }
        
        logger.info(
            "architectures_compared",
            model_ids=model_ids,
            count=len(models),
        )
        
        return ModelComparisonResponse(
            architectures=[_convert_to_response(m) for m in models],
            comparison_table=comparison_table,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("architecture_comparison_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare architectures: {str(e)}",
        )


@router.get(
    "/recommended",
    response_model=ModelListResponse,
    summary="Get recommended model architectures",
    description="Get models with RECOMMENDED badge, sorted by use case priority.",
)
async def get_recommended() -> ModelListResponse:
    """
    Get recommended model architectures.
    
    Returns models with the [RECOMMENDED] badge, which are:
    - Production-ready
    - Well-tested
    - Good balance of accuracy/speed/efficiency
    
    Returns:
        ModelListResponse with recommended models
    """
    try:
        models = get_models_by_badge("RECOMMENDED")
        
        logger.info("recommended_architectures_retrieved", count=len(models))
        
        return ModelListResponse(
            total=len(models),
            architectures=[_convert_to_response(m) for m in models],
        )
    
    except Exception as e:
        logger.error("recommended_architectures_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommended architectures: {str(e)}",
        )


@router.get(
    "/architectures/{model_id}/card",
    response_model=dict,
    summary="Get formatted model card",
    description="Get a human-readable model card with emoji, badges, and ratings.",
)
async def get_model_card(model_id: str) -> dict:
    """
    Get a formatted model card for display.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dict with formatted model card (ready for UI rendering)
    
    Raises:
        404: Model not found
    """
    try:
        model_info = get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model architecture '{model_id}' not found.",
            )
        
        card = model_info_to_dict(model_info)
        
        logger.info("model_card_generated", model_id=model_id)
        
        return card
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("model_card_generation_failed", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate model card: {str(e)}",
        )
