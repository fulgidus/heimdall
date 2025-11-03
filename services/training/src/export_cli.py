#!/usr/bin/env python3
"""
CLI tool for exporting/importing Heimdall model bundles.

Usage:
    # Export a model
    python export_cli.py export <model_id> --output model.heimdall
    
    # Export with options
    python export_cli.py export <model_id> \
        --output model.heimdall \
        --no-config \
        --no-samples \
        --description "Production model v1.2"
    
    # Import a model
    python export_cli.py import model.heimdall
    
    # List available models
    python export_cli.py list
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from export.heimdall_format import HeimdallExporter, HeimdallImporter, HeimdallBundle
from storage.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def export_model(
    model_id: str,
    output_path: Path,
    include_config: bool = True,
    include_metrics: bool = True,
    include_normalization: bool = True,
    include_samples: bool = True,
    num_samples: int = 5,
    description: Optional[str] = None
):
    """Export a model to .heimdall bundle."""
    import boto3
    import os
    
    try:
        # Initialize database and storage
        db_manager = get_db_manager()
        
        # Create MinIO client
        minio_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            region_name='us-east-1'
        )
        
        exporter = HeimdallExporter(db_manager, minio_client)
        
        logger.info(f"Exporting model {model_id}...")
        
        # Create bundle
        bundle = exporter.export_model(
            model_id=model_id,
            include_config=include_config,
            include_metrics=include_metrics,
            include_normalization=include_normalization,
            include_samples=include_samples,
            num_samples=num_samples,
            description=description
        )
        
        # Save to file
        exporter.save_bundle(bundle, str(output_path))
        
        # Print summary
        bundle_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Export successful!")
        logger.info(f"  Bundle ID: {bundle.bundle_metadata.bundle_id}")
        logger.info(f"  Model: {bundle.model.model_name} v{bundle.model.version}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Size: {bundle_size_mb:.2f} MB")
        logger.info(f"  Components:")
        logger.info(f"    - ONNX model: ✓")
        logger.info(f"    - Config: {'✓' if bundle.training_config else '✗'}")
        logger.info(f"    - Metrics: {'✓' if bundle.performance_metrics else '✗'}")
        logger.info(f"    - Normalization: {'✓' if bundle.normalization_stats else '✗'}")
        logger.info(f"    - Samples: {'✓' if bundle.sample_predictions else '✗'} "
                   f"({len(bundle.sample_predictions) if bundle.sample_predictions else 0} predictions)")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Export failed: {e}", exc_info=True)
        return 1


def import_model(bundle_path: Path):
    """Import a model from .heimdall bundle."""
    import boto3
    import os
    
    try:
        db_manager = get_db_manager()
        
        # Create MinIO client
        minio_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            region_name='us-east-1'
        )
        
        importer = HeimdallImporter(db_manager, minio_client)
        
        logger.info(f"Importing model from {bundle_path}...")
        
        # Load and validate bundle
        with open(bundle_path, 'rb') as f:
            bundle_json = f.read()
        
        bundle = HeimdallBundle.model_validate_json(bundle_json)
        
        logger.info(f"  Bundle ID: {bundle.bundle_metadata.bundle_id}")
        logger.info(f"  Model: {bundle.model.model_name} v{bundle.model.version}")
        logger.info(f"  Format version: {bundle.format_version}")
        
        # Import
        result = importer.import_model(bundle)
        model_id = result["model_id"]
        
        logger.info(f"✓ Import successful!")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  Model registered in database")
        logger.info(f"  ONNX uploaded to MinIO")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}", exc_info=True)
        return 1


def list_models():
    """List all available models."""
    from sqlalchemy import text
    
    try:
        db_manager = get_db_manager()
        
        query = text("""
        SELECT 
            m.id,
            m.model_name,
            COALESCE(m.version, 1) as version,
            m.accuracy_meters,
            m.created_at,
            m.onnx_model_location
        FROM heimdall.models m
        ORDER BY m.created_at DESC
        """)
        
        with db_manager.get_session() as session:
            result = session.execute(query).fetchall()
        
        if not result:
            logger.info("No models found in database")
            return 0
        
        logger.info(f"Found {len(result)} models:")
        logger.info("")
        logger.info(f"{'ID':<38}  {'Model Name':<35}  {'Version':<8}  {'Accuracy (m)':<12}  {'Created':<20}")
        logger.info("-" * 130)
        
        for row in result:
            model_id = str(row[0])
            model_name = (row[1] or "Unknown")[:35]
            version = str(row[2])
            accuracy = f"{row[3]:.2f}" if row[3] else "N/A"
            created = row[4].strftime("%Y-%m-%d %H:%M") if row[4] else "N/A"
            
            logger.info(f"{model_id}  {model_name:<35}  {version:<8}  {accuracy:<12}  {created:<20}")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Failed to list models: {e}", exc_info=True)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Heimdall Model Export/Import CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model to .heimdall bundle")
    export_parser.add_argument("model_id", help="UUID of model to export")
    export_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .heimdall file path"
    )
    export_parser.add_argument(
        "--no-config",
        action="store_false",
        dest="include_config",
        help="Exclude training configuration"
    )
    export_parser.add_argument(
        "--no-metrics",
        action="store_false",
        dest="include_metrics",
        help="Exclude performance metrics"
    )
    export_parser.add_argument(
        "--no-normalization",
        action="store_false",
        dest="include_normalization",
        help="Exclude normalization stats"
    )
    export_parser.add_argument(
        "--no-samples",
        action="store_false",
        dest="include_samples",
        help="Exclude sample predictions"
    )
    export_parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=5,
        help="Number of sample predictions (default: 5)"
    )
    export_parser.add_argument(
        "-d", "--description",
        help="Optional description for the bundle"
    )
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a model from .heimdall bundle")
    import_parser.add_argument("bundle_path", type=Path, help="Path to .heimdall bundle file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all available models")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "export":
        return export_model(
            model_id=args.model_id,
            output_path=args.output,
            include_config=args.include_config,
            include_metrics=args.include_metrics,
            include_normalization=args.include_normalization,
            include_samples=args.include_samples,
            num_samples=args.num_samples,
            description=args.description
        )
    elif args.command == "import":
        return import_model(args.bundle_path)
    elif args.command == "list":
        return list_models()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
