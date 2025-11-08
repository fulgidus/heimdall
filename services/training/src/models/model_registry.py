"""
ML Model Architecture Registry for Heimdall RF Localization.

This module provides a comprehensive registry of all available ML architectures
for RF source localization, with detailed metadata for informed model selection.

Features:
- 13 registered architectures (spectrogram, IQ-raw, hybrid, transformer, temporal, ensemble, multi-modal)
- Performance metrics (accuracy, speed, memory)
- Visual indicators (emoji, badges, star ratings)
- Query and comparison utilities
- REST API integration ready

Architecture Categories:
- 🖼️  Spectrogram-based: Process mel-spectrograms from IQ data
- 📡 IQ-Raw: Process raw I/Q samples directly
- 🔬 Hybrid: Combine CNN encoders with Transformer aggregation
- 🧠 Transformer: Vision Transformer architectures
- 🌊 Temporal: Temporal Convolutional Networks (TCN/WaveNet)
- 🧮 Feature-based: MLP on extracted features (triangulation)
- 👁️  Multi-Modal: Combine IQ raw + extracted features + geometry
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import structlog

logger = structlog.get_logger(__name__)

# Type aliases for better type hints
DataType = Literal["spectrogram", "iq_raw", "features", "hybrid", "multi_modal"]
ArchitectureType = Literal["cnn", "transformer", "tcn", "hybrid", "mlp", "multi_modal"]
Badge = Literal["RECOMMENDED", "MAXIMUM_ACCURACY", "FASTEST", "BEST_RATIO", "BASELINE", "EXPERIMENTAL", "LIGHTWEIGHT", "FLAGSHIP"]


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model architecture."""
    
    # Accuracy metrics
    expected_error_min_m: float  # Minimum expected localization error (meters)
    expected_error_max_m: float  # Maximum expected localization error (meters)
    accuracy_stars: int  # 1-5 star rating (5 = best)
    
    # Speed metrics
    inference_time_min_ms: float  # Minimum inference time (milliseconds)
    inference_time_max_ms: float  # Maximum inference time (milliseconds)
    speed_stars: int  # 1-5 star rating (5 = fastest)
    
    # Memory metrics
    parameters_millions: float  # Model parameters in millions
    vram_training_gb: float  # VRAM required for training (GB)
    vram_inference_gb: float  # VRAM required for inference (GB)
    efficiency_stars: int  # 1-5 star rating (5 = most efficient)
    
    # Visual indicators (emoji)
    speed_emoji: str  # 🚀 (ultra fast) → 🏃 (fast) → 🚶 (medium) → 🐢 (slow)
    memory_emoji: str  # 📦 (lightweight) → 📚 (medium) → 🏪 (heavy) → 🏢 (very heavy)
    accuracy_emoji: str  # 💎 (premium) → 💍 (excellent) → ⚪ (good) → ⚫ (basic)


@dataclass
class ModelArchitectureInfo:
    """Complete metadata for a model architecture."""
    
    # Identity
    id: str  # Unique identifier (e.g., "iq_transformer")
    display_name: str  # Human-readable name (e.g., "IQ Transformer")
    description: str  # Short description (1-2 sentences)
    long_description: str  # Detailed description with architecture details
    
    # Classification
    data_type: DataType  # Input data type
    architecture_type: ArchitectureType  # Architecture family
    architecture_emoji: str  # Visual category indicator
    
    # Performance
    performance: PerformanceMetrics
    
    # Badges and highlights
    badges: list[Badge] = field(default_factory=list)
    
    # Use cases and recommendations
    best_for: list[str] = field(default_factory=list)  # Ideal use cases
    not_recommended_for: list[str] = field(default_factory=list)  # When to avoid
    
    # Technical details
    backbone: Optional[str] = None  # Backbone architecture (e.g., "ResNet-18")
    pretrained_weights: Optional[str] = None  # Pretrained weights source
    input_shape: Optional[tuple] = None  # Expected input shape
    output_shape: Optional[tuple] = None  # Expected output shape
    
    # Training considerations
    training_difficulty: Literal["easy", "medium", "hard"] = "medium"
    convergence_epochs: int = 50  # Typical epochs to convergence
    recommended_batch_size: int = 32
    
    # References
    paper_url: Optional[str] = None
    implementation_file: str = ""  # Python file path


# ============================================================================
# REGISTRY: All 13 Model Architectures
# ============================================================================

MODEL_REGISTRY: dict[str, ModelArchitectureInfo] = {
    # ------------------------------------------------------------------------
    # SPECTROGRAM-BASED MODELS (🖼️)
    # ------------------------------------------------------------------------
    "localization_net_convnext_large": ModelArchitectureInfo(
        id="localization_net_convnext_large",
        display_name="LocalizationNet (ConvNeXt-Large)",
        description="ConvNeXt-Large backbone with dual-head output for position and uncertainty.",
        long_description=(
            "Modern CNN architecture using ConvNeXt-Large (200M params, 88.6% ImageNet top-1 accuracy) "
            "as backbone. Processes mel-spectrograms (3-channel: I, Q, magnitude) with 128 frequency bins "
            "and 32 time frames. Features dual heads for position prediction and uncertainty estimation. "
            "Excellent for production deployments with good balance of accuracy and speed. "
            "Recommended as baseline for spectrogram-based approaches."
        ),
        data_type="spectrogram",
        architecture_type="cnn",
        architecture_emoji="🖼️",
        performance=PerformanceMetrics(
            expected_error_min_m=22.0,
            expected_error_max_m=30.0,
            accuracy_stars=4,
            inference_time_min_ms=40.0,
            inference_time_max_ms=50.0,
            speed_stars=3,
            parameters_millions=200.0,
            vram_training_gb=8.0,
            vram_inference_gb=2.0,
            efficiency_stars=2,
            speed_emoji="🚶",
            memory_emoji="🏢",
            accuracy_emoji="💍",
        ),
        badges=["BASELINE"],
        best_for=[
            "Production deployments with balanced requirements",
            "When you have pretrained weights available",
            "Stable training with reliable convergence",
            "Good starting point for experimentation",
        ],
        not_recommended_for=[
            "Edge devices with limited memory",
            "Ultra-low latency requirements (<30ms)",
            "When training data is very limited (<1000 samples)",
        ],
        backbone="ConvNeXt-Large",
        pretrained_weights="ImageNet1K_V1",
        input_shape=(None, 3, 128, 32),  # (batch, channels, freq_bins, time_frames)
        output_shape=(None, 2),  # (batch, [lat, lon])
        training_difficulty="easy",
        convergence_epochs=30,
        recommended_batch_size=32,
        paper_url="https://arxiv.org/abs/2201.03545",
        implementation_file="models/localization_net.py",
    ),
    
    "localization_net_vit": ModelArchitectureInfo(
        id="localization_net_vit",
        display_name="LocalizationNet (Vision Transformer)",
        description="Vision Transformer adapted for mel-spectrogram processing with patch embeddings.",
        long_description=(
            "Vision Transformer (ViT) architecture adapted for RF localization. Splits mel-spectrograms "
            "into patches (16x16), embeds them with learned positional encodings, and processes through "
            "12-layer transformer encoder. Excellent self-attention capabilities for capturing long-range "
            "dependencies in spectrograms. Best accuracy among spectrogram models but slower inference. "
            "Requires more training data than CNNs (recommended >5000 samples)."
        ),
        data_type="spectrogram",
        architecture_type="transformer",
        architecture_emoji="🖼️",
        performance=PerformanceMetrics(
            expected_error_min_m=18.0,
            expected_error_max_m=25.0,
            accuracy_stars=5,
            inference_time_min_ms=80.0,
            inference_time_max_ms=120.0,
            speed_stars=2,
            parameters_millions=86.0,
            vram_training_gb=10.0,
            vram_inference_gb=2.5,
            efficiency_stars=2,
            speed_emoji="🐢",
            memory_emoji="🏢",
            accuracy_emoji="💎",
        ),
        badges=["EXPERIMENTAL"],
        best_for=[
            "Maximum accuracy requirements",
            "Large training datasets (>5000 samples)",
            "Research and experimentation",
            "When inference time is not critical",
        ],
        not_recommended_for=[
            "Small datasets (<2000 samples)",
            "Real-time applications with strict latency",
            "Limited GPU memory scenarios",
        ],
        backbone="ViT-Base",
        pretrained_weights="ImageNet21k",
        input_shape=(None, 3, 128, 32),
        output_shape=(None, 2),
        training_difficulty="hard",
        convergence_epochs=80,
        recommended_batch_size=16,
        paper_url="https://arxiv.org/abs/2010.11929",
        implementation_file="models/localization_net.py",
    ),
    
    # ------------------------------------------------------------------------
    # IQ-RAW CNN MODELS (📡)
    # ------------------------------------------------------------------------
    "iq_resnet18": ModelArchitectureInfo(
        id="iq_resnet18",
        display_name="IQ ResNet-18",
        description="ResNet-18 adapted for raw IQ samples with attention aggregation over receivers.",
        long_description=(
            "Lightweight ResNet-18 backbone processing raw I/Q samples (2-channel: I, Q) from multiple "
            "receivers. Features per-receiver CNN encoder with shared weights, multi-head attention "
            "aggregation over receiver embeddings, and dual-head output. Fast inference and low memory "
            "footprint make it ideal for edge deployments. Good baseline for IQ-raw approaches."
        ),
        data_type="iq_raw",
        architecture_type="cnn",
        architecture_emoji="📡",
        performance=PerformanceMetrics(
            expected_error_min_m=30.0,
            expected_error_max_m=40.0,
            accuracy_stars=3,
            inference_time_min_ms=30.0,
            inference_time_max_ms=50.0,
            speed_stars=4,
            parameters_millions=15.0,
            vram_training_gb=4.0,
            vram_inference_gb=1.0,
            efficiency_stars=4,
            speed_emoji="🏃",
            memory_emoji="📦",
            accuracy_emoji="⚪",
        ),
        badges=["BASELINE"],
        best_for=[
            "Edge devices with limited resources",
            "Fast prototyping and experimentation",
            "When working with raw IQ data",
            "Variable number of receivers (5-10)",
        ],
        not_recommended_for=[
            "Maximum accuracy requirements",
            "When spectrograms are available",
        ],
        backbone="ResNet-18",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),  # (batch, max_receivers, [I,Q], seq_len)
        output_shape=(None, 2),
        training_difficulty="easy",
        convergence_epochs=40,
        recommended_batch_size=64,
        implementation_file="models/iq_cnn_models.py",
    ),
    
    "iq_resnet50": ModelArchitectureInfo(
        id="iq_resnet50",
        display_name="IQ ResNet-50",
        description="Deeper ResNet-50 for IQ processing with improved feature extraction.",
        long_description=(
            "Mid-depth ResNet-50 backbone (25M params) for raw IQ samples. Provides better feature "
            "extraction than ResNet-18 with moderate computational cost increase. Good balance between "
            "accuracy and efficiency for production deployments. Recommended when you need better "
            "accuracy than ResNet-18 but can't afford the memory of ResNet-101."
        ),
        data_type="iq_raw",
        architecture_type="cnn",
        architecture_emoji="📡",
        performance=PerformanceMetrics(
            expected_error_min_m=25.0,
            expected_error_max_m=32.0,
            accuracy_stars=4,
            inference_time_min_ms=50.0,
            inference_time_max_ms=80.0,
            speed_stars=3,
            parameters_millions=25.0,
            vram_training_gb=6.0,
            vram_inference_gb=1.5,
            efficiency_stars=3,
            speed_emoji="🚶",
            memory_emoji="📚",
            accuracy_emoji="💍",
        ),
        badges=[],
        best_for=[
            "Production deployments requiring better accuracy",
            "When training data is abundant (>3000 samples)",
            "Balance between speed and accuracy",
        ],
        not_recommended_for=[
            "Edge devices with <4GB VRAM",
            "Ultra-low latency requirements",
        ],
        backbone="ResNet-50",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="medium",
        convergence_epochs=50,
        recommended_batch_size=32,
        implementation_file="models/iq_cnn_models.py",
    ),
    
    "iq_resnet101": ModelArchitectureInfo(
        id="iq_resnet101",
        display_name="IQ ResNet-101",
        description="Very deep ResNet-101 for maximum IQ feature extraction capacity.",
        long_description=(
            "Very deep ResNet-101 backbone (44M params) for raw IQ samples. Provides excellent feature "
            "extraction with the deepest CNN architecture in the IQ family. Best accuracy among pure CNN "
            "IQ models but requires significant GPU memory. Recommended only when you have abundant "
            "training data and GPU resources."
        ),
        data_type="iq_raw",
        architecture_type="cnn",
        architecture_emoji="📡",
        performance=PerformanceMetrics(
            expected_error_min_m=22.0,
            expected_error_max_m=28.0,
            accuracy_stars=4,
            inference_time_min_ms=80.0,
            inference_time_max_ms=120.0,
            speed_stars=2,
            parameters_millions=44.0,
            vram_training_gb=8.0,
            vram_inference_gb=2.0,
            efficiency_stars=2,
            speed_emoji="🚶",
            memory_emoji="🏪",
            accuracy_emoji="💍",
        ),
        badges=[],
        best_for=[
            "Maximum accuracy with CNN architectures",
            "Large datasets (>5000 samples)",
            "High-end GPU deployments (RTX 3090+)",
        ],
        not_recommended_for=[
            "Limited GPU memory (<8GB)",
            "Fast inference requirements",
            "Small datasets (<2000 samples)",
        ],
        backbone="ResNet-101",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="medium",
        convergence_epochs=60,
        recommended_batch_size=16,
        implementation_file="models/iq_cnn_models.py",
    ),
    
    "iq_vggnet": ModelArchitectureInfo(
        id="iq_vggnet",
        display_name="IQ VGG-Style",
        description="Simple VGG-style CNN for fast IQ processing with minimal overhead.",
        long_description=(
            "Lightweight VGG-style architecture (12M params) for raw IQ samples. Uses stacked 3x3 "
            "convolutions with max pooling for simple yet effective feature extraction. Fastest inference "
            "among all IQ models with minimal memory footprint. Ideal for edge devices and rapid "
            "prototyping. Trades some accuracy for speed and simplicity."
        ),
        data_type="iq_raw",
        architecture_type="cnn",
        architecture_emoji="📡",
        performance=PerformanceMetrics(
            expected_error_min_m=35.0,
            expected_error_max_m=45.0,
            accuracy_stars=2,
            inference_time_min_ms=25.0,
            inference_time_max_ms=40.0,
            speed_stars=5,
            parameters_millions=12.0,
            vram_training_gb=3.0,
            vram_inference_gb=0.8,
            efficiency_stars=5,
            speed_emoji="🚀",
            memory_emoji="📦",
            accuracy_emoji="⚪",
        ),
        badges=["FASTEST"],
        best_for=[
            "Ultra-low latency requirements (<30ms)",
            "Edge devices and embedded systems",
            "Fast prototyping and baseline comparisons",
            "When training resources are limited",
        ],
        not_recommended_for=[
            "High accuracy requirements",
            "Complex RF environments with interference",
        ],
        backbone="VGG-Style",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="easy",
        convergence_epochs=30,
        recommended_batch_size=128,
        implementation_file="models/iq_cnn_models.py",
    ),
    
    # ------------------------------------------------------------------------
    # TRANSFORMER MODELS (🧠)
    # ------------------------------------------------------------------------
    "iq_transformer": ModelArchitectureInfo(
        id="iq_transformer",
        display_name="IQ Transformer",
        description="Vision Transformer adapted for raw IQ sequences with patch embeddings.",
        long_description=(
            "Pure transformer architecture (ViT-style) for raw IQ samples. Processes IQ sequences as "
            "1D patches with learned positional encodings. 12-layer transformer encoder with 12 attention "
            "heads provides outstanding modeling capacity. Best overall accuracy among all models but "
            "slowest inference. Requires large training datasets (>5000 samples) and significant GPU "
            "memory. Recommended for research and maximum accuracy scenarios."
        ),
        data_type="iq_raw",
        architecture_type="transformer",
        architecture_emoji="🧠",
        performance=PerformanceMetrics(
            expected_error_min_m=10.0,
            expected_error_max_m=18.0,
            accuracy_stars=5,
            inference_time_min_ms=100.0,
            inference_time_max_ms=200.0,
            speed_stars=1,
            parameters_millions=80.0,
            vram_training_gb=12.0,
            vram_inference_gb=3.0,
            efficiency_stars=1,
            speed_emoji="🐢",
            memory_emoji="🏢",
            accuracy_emoji="💎",
        ),
        badges=["MAXIMUM_ACCURACY", "EXPERIMENTAL"],
        best_for=[
            "Absolute maximum accuracy requirements",
            "Research and academic papers",
            "Large datasets (>5000 samples)",
            "High-end GPU infrastructure",
        ],
        not_recommended_for=[
            "Production deployments with latency constraints",
            "Small datasets (<2000 samples)",
            "Limited GPU resources (<12GB VRAM)",
            "Edge devices",
        ],
        backbone="ViT-Base",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="hard",
        convergence_epochs=100,
        recommended_batch_size=8,
        paper_url="https://arxiv.org/abs/2010.11929",
        implementation_file="models/iq_transformer.py",
    ),
    
    # ------------------------------------------------------------------------
    # TEMPORAL CONVOLUTIONAL NETWORKS (🌊)
    # ------------------------------------------------------------------------
    "iq_wavenet": ModelArchitectureInfo(
        id="iq_wavenet",
        display_name="IQ WaveNet/TCN",
        description="Temporal Convolutional Network with dilated causal convolutions for IQ sequences.",
        long_description=(
            "Temporal Convolutional Network (TCN) inspired by WaveNet architecture. Uses dilated causal "
            "convolutions with exponentially increasing dilation rates to capture long-range temporal "
            "dependencies in IQ sequences. Excellent for modeling time-varying RF propagation effects. "
            "50M parameters with residual connections provide strong modeling capacity. Outstanding "
            "accuracy with reasonable inference time. Recommended when temporal dynamics are important."
        ),
        data_type="iq_raw",
        architecture_type="tcn",
        architecture_emoji="🌊",
        performance=PerformanceMetrics(
            expected_error_min_m=20.0,
            expected_error_max_m=28.0,
            accuracy_stars=5,
            inference_time_min_ms=50.0,
            inference_time_max_ms=70.0,
            speed_stars=3,
            parameters_millions=50.0,
            vram_training_gb=8.0,
            vram_inference_gb=2.0,
            efficiency_stars=2,
            speed_emoji="🚶",
            memory_emoji="🏪",
            accuracy_emoji="💎",
        ),
        badges=[],
        best_for=[
            "Time-varying RF environments",
            "Capturing temporal propagation effects",
            "Mobile/moving transmitters",
            "When temporal context matters",
        ],
        not_recommended_for=[
            "Static transmitter localization",
            "Edge devices with limited memory",
        ],
        backbone="WaveNet-TCN",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="medium",
        convergence_epochs=60,
        recommended_batch_size=16,
        paper_url="https://arxiv.org/abs/1803.01271",
        implementation_file="models/iq_wavenet.py",
    ),
    
    # ------------------------------------------------------------------------
    # EFFICIENT ARCHITECTURES (⚡)
    # ------------------------------------------------------------------------
    "iq_efficientnet_b4": ModelArchitectureInfo(
        id="iq_efficientnet_b4",
        display_name="IQ EfficientNet-B4",
        description="EfficientNet-B4 with compound scaling for optimal accuracy/efficiency balance.",
        long_description=(
            "EfficientNet-B4 (22M params) adapted for raw IQ samples. Uses compound scaling to balance "
            "network depth, width, and resolution. Mobile inverted bottleneck convolutions (MBConv) with "
            "squeeze-and-excitation provide excellent efficiency. Best accuracy-to-parameter ratio among "
            "all models. Recommended for production deployments when you need strong accuracy without "
            "excessive memory requirements."
        ),
        data_type="iq_raw",
        architecture_type="cnn",
        architecture_emoji="📡",
        performance=PerformanceMetrics(
            expected_error_min_m=22.0,
            expected_error_max_m=30.0,
            accuracy_stars=4,
            inference_time_min_ms=40.0,
            inference_time_max_ms=60.0,
            speed_stars=4,
            parameters_millions=22.0,
            vram_training_gb=5.0,
            vram_inference_gb=1.2,
            efficiency_stars=5,
            speed_emoji="🏃",
            memory_emoji="📚",
            accuracy_emoji="💍",
        ),
        badges=["BEST_RATIO"],
        best_for=[
            "Production deployments with balanced needs",
            "Best accuracy per parameter",
            "Mid-range GPU hardware",
            "When efficiency matters",
        ],
        not_recommended_for=[
            "When maximum accuracy is required at any cost",
            "Very limited GPU memory (<3GB)",
        ],
        backbone="EfficientNet-B4",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="medium",
        convergence_epochs=50,
        recommended_batch_size=32,
        paper_url="https://arxiv.org/abs/1905.11946",
        implementation_file="models/iq_efficientnet.py",
    ),
    
    # ------------------------------------------------------------------------
    # HYBRID MODELS (🔬)
    # ------------------------------------------------------------------------
    "iq_hybrid": ModelArchitectureInfo(
        id="iq_hybrid",
        display_name="IQ HybridNet (CNN+Transformer)",
        description="DETR-style hybrid combining ResNet-50 encoder with Transformer aggregation.",
        long_description=(
            "Hybrid architecture combining CNN feature extraction with Transformer aggregation. Uses "
            "ResNet-50 to encode each receiver's IQ samples, then applies 6-layer transformer to aggregate "
            "multi-receiver information with learned attention. Inspired by DETR (DEtection TRansformer). "
            "Outstanding accuracy (±12-20m) with reasonable inference time. Best overall choice for "
            "production deployments when accuracy is critical. Recommended as primary model."
        ),
        data_type="hybrid",
        architecture_type="hybrid",
        architecture_emoji="🔬",
        performance=PerformanceMetrics(
            expected_error_min_m=12.0,
            expected_error_max_m=20.0,
            accuracy_stars=5,
            inference_time_min_ms=120.0,
            inference_time_max_ms=180.0,
            speed_stars=2,
            parameters_millions=70.0,
            vram_training_gb=10.0,
            vram_inference_gb=2.5,
            efficiency_stars=2,
            speed_emoji="🚶",
            memory_emoji="🏪",
            accuracy_emoji="💎",
        ),
        badges=["RECOMMENDED"],
        best_for=[
            "Production deployments requiring high accuracy",
            "Multi-receiver scenarios (5-10 receivers)",
            "Complex RF environments",
            "When both speed and accuracy matter",
        ],
        not_recommended_for=[
            "Ultra-low latency requirements (<50ms)",
            "Edge devices with <6GB VRAM",
            "Small datasets (<2000 samples)",
        ],
        backbone="ResNet-50 + Transformer",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),
        output_shape=(None, 2),
        training_difficulty="medium",
        convergence_epochs=50,
        recommended_batch_size=16,
        paper_url="https://arxiv.org/abs/2005.12872",
        implementation_file="models/hybrid_models.py",
    ),
    
    # ------------------------------------------------------------------------
    # MULTI-MODAL MODELS (👁️)
    # ------------------------------------------------------------------------
    "heimdall_net": ModelArchitectureInfo(
        id="heimdall_net",
        display_name="HeimdallNet 👁️ (Multi-Modal Fusion)",
        description="Adaptive multi-receiver network combining IQ raw + extracted features + geometry with learnable antenna embeddings.",
        long_description=(
            "Advanced multi-modal architecture named after Heimdall, Norse guardian known for extraordinary "
            "sight. Combines three information sources: (1) raw IQ samples processed by EfficientNet-B2 1D, "
            "(2) extracted RF features (SNR, PSD, freq_offset), and (3) geometric relationships between receivers. "
            "Key innovation: learnable per-receiver embeddings capture unique antenna characteristics (gain patterns, "
            "directivity, hardware biases) while maintaining permutation invariance via set-based aggregation. "
            "Handles variable receiver count (1-10) with dropout resilience training. Quality-weighted attention "
            "aggregation ensures robust operation with 2-3 active receivers typical in real deployments. "
            "Target accuracy: ±8-15m with 40-60ms inference. Recommended as primary model for production."
        ),
        data_type="multi_modal",
        architecture_type="multi_modal",
        architecture_emoji="👁️",
        performance=PerformanceMetrics(
            expected_error_min_m=8.0,
            expected_error_max_m=15.0,
            accuracy_stars=5,
            inference_time_min_ms=40.0,
            inference_time_max_ms=60.0,
            speed_stars=4,
            parameters_millions=2.2,
            vram_training_gb=3.0,
            vram_inference_gb=0.8,
            efficiency_stars=5,
            speed_emoji="🏃",
            memory_emoji="📦",
            accuracy_emoji="💎",
        ),
        badges=["RECOMMENDED"],
        best_for=[
            "Production deployments with multi-modal data",
            "Variable receiver count (1-10 SDRs)",
            "Unstable receivers with frequent dropouts",
            "Maximum precision (±8-15m target)",
            "Real-world WebSDR scenarios (3/6 online typical)",
            "Capturing antenna-specific characteristics",
        ],
        not_recommended_for=[
            "When only IQ raw is available (use IQ HybridNet)",
            "When only features available (use Triangulation MLP)",
            "Single receiver scenarios",
        ],
        backbone="EfficientNet-B2 1D + Set Attention + Geometry Encoder",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),  # (batch, max_receivers, [I,Q], seq_len)
        output_shape=(None, 2),  # (batch, [lat, lon])
        training_difficulty="medium",
        convergence_epochs=50,
        recommended_batch_size=32,
        implementation_file="models/heimdall_net.py",
    ),
    
    # ------------------------------------------------------------------------
    # FEATURE-BASED MODELS (🧮)
    # ------------------------------------------------------------------------
    "triangulation_model": ModelArchitectureInfo(
        id="triangulation_model",
        display_name="Triangulation MLP",
        description="Lightweight MLP with attention aggregation over extracted features.",
        long_description=(
            "Simple multi-layer perceptron (MLP) operating on extracted RF features (SNR, PSD, frequency "
            "offset, receiver position). Uses multi-head attention to aggregate measurements from multiple "
            "receivers. Extremely fast (<10ms inference) with minimal memory (<1M params). Good for "
            "baseline comparisons and scenarios where raw IQ/spectrograms are unavailable. Limited "
            "accuracy due to reliance on hand-crafted features."
        ),
        data_type="features",
        architecture_type="mlp",
        architecture_emoji="🧮",
        performance=PerformanceMetrics(
            expected_error_min_m=45.0,
            expected_error_max_m=60.0,
            accuracy_stars=2,
            inference_time_min_ms=5.0,
            inference_time_max_ms=10.0,
            speed_stars=5,
            parameters_millions=0.3,
            vram_training_gb=1.0,
            vram_inference_gb=0.2,
            efficiency_stars=5,
            speed_emoji="🚀",
            memory_emoji="📦",
            accuracy_emoji="⚫",
        ),
        badges=["LIGHTWEIGHT", "BASELINE"],
        best_for=[
            "Ultra-fast inference requirements",
            "Embedded systems and IoT devices",
            "When raw IQ data is unavailable",
            "Baseline comparisons",
        ],
        not_recommended_for=[
            "High accuracy requirements",
            "When raw IQ or spectrograms are available",
            "Complex RF propagation scenarios",
        ],
        backbone="MLP + Attention",
        pretrained_weights=None,
        input_shape=(None, 10, 6),  # (batch, max_receivers, [snr, psd, freq, lat, lon, signal])
        output_shape=(None, 2),
        training_difficulty="easy",
        convergence_epochs=20,
        recommended_batch_size=256,
        implementation_file="models/triangulator.py",
    ),
    
    # ------------------------------------------------------------------------
    # ENSEMBLE MODELS (🎯)
    # ------------------------------------------------------------------------
    "localization_ensemble_flagship": ModelArchitectureInfo(
        id="localization_ensemble_flagship",
        display_name="Ensemble Flagship (IQ Transformer + HybridNet + WaveNet)",
        description="Ensemble of top 3 models with learned weighting for ultimate accuracy.",
        long_description=(
            "Ultimate accuracy model combining the best of three worlds: IQ Transformer (±10-18m, "
            "pure attention), IQ HybridNet (±12-20m, CNN+Transformer), and IQ WaveNet/TCN (±20-28m, "
            "temporal modeling). Runs inference through all 3 models in parallel and combines predictions "
            "using learned attention-based ensemble weights. Each model contributes its strengths: "
            "Transformer excels at global dependencies, HybridNet at multi-receiver fusion, and WaveNet "
            "at temporal dynamics. Target accuracy: ±5-12m (68% confidence). Extremely expensive inference "
            "(500-800ms) and massive memory (200M params, 20GB training VRAM). Only recommended for "
            "scenarios where accuracy is paramount and computational resources are abundant. Ideal for "
            "research benchmarks, competitions, and mission-critical deployments."
        ),
        data_type="hybrid",
        architecture_type="hybrid",
        architecture_emoji="🎯",
        performance=PerformanceMetrics(
            expected_error_min_m=5.0,  # Best-in-class accuracy
            expected_error_max_m=12.0,
            accuracy_stars=5,
            inference_time_min_ms=500.0,  # Sum of all 3 models
            inference_time_max_ms=800.0,
            speed_stars=1,
            parameters_millions=200.0,  # 80M + 70M + 50M
            vram_training_gb=20.0,  # Peak memory for all 3 models
            vram_inference_gb=6.0,  # 3.0 + 2.5 + 2.0 (minus sharing)
            efficiency_stars=1,
            speed_emoji="🐌",  # Slowest possible
            memory_emoji="🏢",
            accuracy_emoji="💎💎",  # Double diamond for ultimate accuracy
        ),
        badges=["MAXIMUM_ACCURACY", "EXPERIMENTAL", "FLAGSHIP"],
        best_for=[
            "Absolute maximum accuracy requirements (±5-12m)",
            "Research papers and academic benchmarks",
            "Mission-critical applications (search & rescue, defense)",
            "Competitions and leaderboards",
            "High-end GPU infrastructure (A100/H100 with 40GB+ VRAM)",
            "Offline batch processing with no latency constraints",
        ],
        not_recommended_for=[
            "Production deployments with latency requirements",
            "Real-time applications",
            "Limited GPU resources (<20GB VRAM)",
            "Edge devices and embedded systems",
            "Small datasets (<5000 samples)",
            "Cost-sensitive deployments",
        ],
        backbone="Ensemble: ViT-Base + ResNet-50+Transformer + WaveNet-TCN",
        pretrained_weights=None,
        input_shape=(None, 10, 2, 1024),  # Same as IQ models
        output_shape=(None, 2),
        training_difficulty="hard",
        convergence_epochs=120,  # Ensemble training takes longer
        recommended_batch_size=4,  # Very small due to memory
        paper_url="https://arxiv.org/abs/1906.04569",  # Ensemble methods paper
        implementation_file="models/ensemble_flagship.py",
    ),
}


# ============================================================================
# QUERY AND UTILITY FUNCTIONS
# ============================================================================

def get_model_info(model_id: str) -> ModelArchitectureInfo:
    """
    Get detailed information for a specific model architecture.
    
    Args:
        model_id: Unique model identifier
    
    Returns:
        ModelArchitectureInfo object
    
    Raises:
        KeyError: If model_id not found in registry
    """
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_id}' not found in registry. "
            f"Available models: {available}"
        )
    
    return MODEL_REGISTRY[model_id]


def list_models(
    data_type: Optional[DataType] = None,
    architecture_type: Optional[ArchitectureType] = None,
    min_accuracy_stars: Optional[int] = None,
    min_speed_stars: Optional[int] = None,
    badges: Optional[list[Badge]] = None,
) -> list[ModelArchitectureInfo]:
    """
    List models matching specified criteria.
    
    Args:
        data_type: Filter by input data type
        architecture_type: Filter by architecture family
        min_accuracy_stars: Minimum accuracy rating (1-5)
        min_speed_stars: Minimum speed rating (1-5)
        badges: Filter by badges (must have at least one)
    
    Returns:
        List of matching ModelArchitectureInfo objects
    """
    results = []
    
    for model_info in MODEL_REGISTRY.values():
        # Apply filters
        if data_type and model_info.data_type != data_type:
            continue
        if architecture_type and model_info.architecture_type != architecture_type:
            continue
        if min_accuracy_stars and model_info.performance.accuracy_stars < min_accuracy_stars:
            continue
        if min_speed_stars and model_info.performance.speed_stars < min_speed_stars:
            continue
        if badges and not any(badge in model_info.badges for badge in badges):
            continue
        
        results.append(model_info)
    
    # Sort by accuracy (descending)
    results.sort(key=lambda m: m.performance.accuracy_stars, reverse=True)
    
    return results


def compare_models(model_ids: list[str]) -> dict[str, ModelArchitectureInfo]:
    """
    Compare multiple models side-by-side.
    
    Args:
        model_ids: List of model IDs to compare
    
    Returns:
        Dictionary mapping model_id -> ModelArchitectureInfo
    
    Raises:
        KeyError: If any model_id not found
    """
    comparison = {}
    
    for model_id in model_ids:
        comparison[model_id] = get_model_info(model_id)
    
    return comparison


def get_recommended_model(
    use_case: Literal["accuracy", "speed", "balanced", "edge"] = "balanced"
) -> ModelArchitectureInfo:
    """
    Get recommended model for a specific use case.
    
    Args:
        use_case: Optimization target
            - "accuracy": Maximum localization accuracy
            - "speed": Minimum inference latency
            - "balanced": Best overall for production
            - "edge": Lightweight for edge devices
    
    Returns:
        Recommended ModelArchitectureInfo
    """
    recommendations = {
        "accuracy": "iq_transformer",  # Maximum accuracy
        "speed": "iq_vggnet",  # Fastest inference
        "balanced": "heimdall_net",  # Best overall (multi-modal)
        "edge": "iq_vggnet",  # Lightweight
    }
    
    model_id = recommendations[use_case]
    return get_model_info(model_id)


def get_models_by_badge(badge: Badge) -> list[ModelArchitectureInfo]:
    """
    Get all models with a specific badge.
    
    Args:
        badge: Badge to filter by
    
    Returns:
        List of models with the badge
    """
    return [
        model_info
        for model_info in MODEL_REGISTRY.values()
        if badge in model_info.badges
    ]


def get_all_model_ids() -> list[str]:
    """Get list of all model IDs in registry."""
    return list(MODEL_REGISTRY.keys())


def get_model_count() -> int:
    """Get total number of registered models."""
    return len(MODEL_REGISTRY)


# ============================================================================
# SERIALIZATION FOR REST API
# ============================================================================

def model_info_to_dict(model_info: ModelArchitectureInfo) -> dict:
    """
    Convert ModelArchitectureInfo to dictionary for JSON serialization.
    
    Args:
        model_info: Model architecture info object
    
    Returns:
        Dictionary representation
    """
    return {
        "id": model_info.id,
        "display_name": model_info.display_name,
        "description": model_info.description,
        "long_description": model_info.long_description,
        "data_type": model_info.data_type,
        "architecture_type": model_info.architecture_type,
        "architecture_emoji": model_info.architecture_emoji,
        "performance": {
            "expected_error_min_m": model_info.performance.expected_error_min_m,
            "expected_error_max_m": model_info.performance.expected_error_max_m,
            "accuracy_stars": model_info.performance.accuracy_stars,
            "inference_time_min_ms": model_info.performance.inference_time_min_ms,
            "inference_time_max_ms": model_info.performance.inference_time_max_ms,
            "speed_stars": model_info.performance.speed_stars,
            "parameters_millions": model_info.performance.parameters_millions,
            "vram_training_gb": model_info.performance.vram_training_gb,
            "vram_inference_gb": model_info.performance.vram_inference_gb,
            "efficiency_stars": model_info.performance.efficiency_stars,
            "speed_emoji": model_info.performance.speed_emoji,
            "memory_emoji": model_info.performance.memory_emoji,
            "accuracy_emoji": model_info.performance.accuracy_emoji,
        },
        "badges": model_info.badges,
        "best_for": model_info.best_for,
        "not_recommended_for": model_info.not_recommended_for,
        "backbone": model_info.backbone,
        "pretrained_weights": model_info.pretrained_weights,
        "input_shape": model_info.input_shape,
        "output_shape": model_info.output_shape,
        "training_difficulty": model_info.training_difficulty,
        "convergence_epochs": model_info.convergence_epochs,
        "recommended_batch_size": model_info.recommended_batch_size,
        "paper_url": model_info.paper_url,
        "implementation_file": model_info.implementation_file,
    }


# Initialize logging
logger.info(
    "model_registry_loaded",
    total_models=get_model_count(),
    data_types=list(set(m.data_type for m in MODEL_REGISTRY.values())),
    architecture_types=list(set(m.architecture_type for m in MODEL_REGISTRY.values())),
)
