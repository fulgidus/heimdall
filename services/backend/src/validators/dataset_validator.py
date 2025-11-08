"""
Dataset validator for detecting orphaned IQ data and database features.

This module provides functionality to validate the integrity of synthetic datasets
by checking for mismatches between:
1. IQ data files in MinIO storage
2. Feature records in the measurement_features database table
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Dataset health status indicators."""
    
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"  # Minor issues (< 5% orphaned)
    CRITICAL = "critical"  # Major issues (>= 5% orphaned)


@dataclass
class ValidationIssue:
    """Individual validation issue found in dataset."""
    
    issue_type: str  # "orphaned_iq", "orphaned_feature", "missing_iq", "corrupted_iq"
    path: str  # IQ file path or feature ID
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    
    dataset_id: UUID
    health_status: HealthStatus
    validated_at: datetime
    
    # Counts
    total_features: int
    total_iq_files: int
    orphaned_iq_files: int
    orphaned_features: int
    
    # Issues list
    issues: list[ValidationIssue]
    
    # Summary
    summary: str
    
    @property
    def is_healthy(self) -> bool:
        """Check if dataset is healthy."""
        return self.health_status == HealthStatus.HEALTHY
    
    @property
    def orphan_percentage(self) -> float:
        """Calculate percentage of orphaned items."""
        total_items = self.total_features + self.total_iq_files
        if total_items == 0:
            return 0.0
        orphaned = self.orphaned_iq_files + self.orphaned_features
        return (orphaned / total_items) * 100
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "dataset_id": str(self.dataset_id),
            "health_status": self.health_status.value,
            "validated_at": self.validated_at.isoformat(),
            "total_features": self.total_features,
            "total_iq_files": self.total_iq_files,
            "orphaned_iq_files": self.orphaned_iq_files,
            "orphaned_features": self.orphaned_features,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "path": issue.path,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
            "summary": self.summary,
            "orphan_percentage": round(self.orphan_percentage, 2),
        }


class DatasetValidator:
    """Validates dataset integrity and detects orphaned data."""
    
    def __init__(self, db_manager, minio_client):
        """
        Initialize dataset validator.
        
        Args:
            db_manager: Database manager for PostgreSQL queries
            minio_client: MinIO client for S3 operations
        """
        self.db_manager = db_manager
        self.minio_client = minio_client
        logger.info("DatasetValidator initialized")
    
    def validate_dataset(self, dataset_id: UUID) -> ValidationReport:
        """
        Validate a synthetic dataset for integrity issues.
        
        Checks for:
        1. IQ files in MinIO without corresponding database records
        2. Database records without corresponding IQ files
        
        Supports both dataset types:
        - iq_raw: validates synthetic_iq_samples against MinIO files
        - feature_based: validates measurement_features against MinIO files
        
        Args:
            dataset_id: UUID of the dataset to validate
            
        Returns:
            ValidationReport with all issues found
        """
        logger.info(f"Starting validation for dataset {dataset_id}")
        
        try:
            # Get dataset type to determine which table to query
            dataset_type = self._get_dataset_type(dataset_id)
            logger.info(f"Dataset type: {dataset_type}")
            
            # Get all records from database (based on dataset type)
            if dataset_type == 'iq_raw':
                features_map = self._get_iq_raw_samples(dataset_id)
            else:
                features_map = self._get_dataset_features(dataset_id)
            logger.info(f"Found {len(features_map)} records in database")
            
            # Get all IQ files from MinIO
            iq_files_set = self._get_dataset_iq_files(dataset_id)
            logger.info(f"Found {len(iq_files_set)} IQ files in MinIO")
            
            # Find orphaned IQ files (in MinIO but not in database)
            orphaned_iq_files = []
            for iq_path in iq_files_set:
                if iq_path not in features_map:
                    orphaned_iq_files.append(ValidationIssue(
                        issue_type="orphaned_iq",
                        path=iq_path,
                        details="IQ file exists in MinIO but no corresponding feature record in database"
                    ))
            
            # Find orphaned features (in database but IQ file missing in MinIO)
            orphaned_features = []
            for iq_path, feature_id in features_map.items():
                if iq_path not in iq_files_set:
                    orphaned_features.append(ValidationIssue(
                        issue_type="orphaned_feature",
                        path=iq_path,
                        details=f"Feature record {feature_id} exists but IQ file missing in MinIO"
                    ))
            
            # Combine all issues
            all_issues = orphaned_iq_files + orphaned_features
            
            # Determine health status
            total_items = len(features_map) + len(iq_files_set)
            orphan_count = len(orphaned_iq_files) + len(orphaned_features)
            
            if orphan_count == 0:
                health_status = HealthStatus.HEALTHY
                summary = "Dataset is healthy - all IQ files and features are properly matched"
            else:
                orphan_pct = (orphan_count / total_items * 100) if total_items > 0 else 0
                if orphan_pct < 5.0:
                    health_status = HealthStatus.WARNING
                    summary = f"Dataset has minor issues: {orphan_count} orphaned items ({orphan_pct:.1f}%)"
                else:
                    health_status = HealthStatus.CRITICAL
                    summary = f"Dataset has major issues: {orphan_count} orphaned items ({orphan_pct:.1f}%)"
            
            report = ValidationReport(
                dataset_id=dataset_id,
                health_status=health_status,
                validated_at=datetime.utcnow(),
                total_features=len(features_map),
                total_iq_files=len(iq_files_set),
                orphaned_iq_files=len(orphaned_iq_files),
                orphaned_features=len(orphaned_features),
                issues=all_issues,
                summary=summary,
            )
            
            logger.info(
                f"Validation complete for dataset {dataset_id}: "
                f"{health_status.value} - {len(all_issues)} issues found"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating dataset {dataset_id}: {e}", exc_info=True)
            raise
    
    def _get_dataset_type(self, dataset_id: UUID) -> str:
        """Get dataset type from database."""
        from sqlalchemy import text
        
        with self.db_manager.get_session() as session:
            query = text("SELECT dataset_type FROM heimdall.synthetic_datasets WHERE id = :dataset_id")
            result = session.execute(query, {"dataset_id": str(dataset_id)}).fetchone()
            return result[0] if result else "feature_based"
    
    def _get_iq_raw_samples(self, dataset_id: UUID) -> dict[str, str]:
        """
        Get all IQ raw sample records for a dataset from database.
        
        Returns:
            Dictionary mapping IQ file paths to sample IDs
        """
        from sqlalchemy import text
        import json
        
        with self.db_manager.get_session() as session:
            query = text("""
                SELECT id, sample_idx, iq_storage_paths
                FROM heimdall.synthetic_iq_samples
                WHERE dataset_id = :dataset_id
            """)
            
            result = session.execute(query, {"dataset_id": str(dataset_id)})
            
            # Build map: iq_path -> sample_id
            samples_map = {}
            for row in result:
                sample_id = str(row[0])
                sample_idx = row[1]
                iq_storage_paths = row[2] if isinstance(row[2], dict) else json.loads(row[2])
                
                # Add all IQ paths for this sample
                for rx_id, iq_path in iq_storage_paths.items():
                    # Normalize path
                    iq_path = self._normalize_s3_path(iq_path)
                    samples_map[iq_path] = sample_id
            
            return samples_map
    
    def _get_dataset_features(self, dataset_id: UUID) -> dict[str, str]:
        """
        Get all feature records for a dataset from database.
        
        Returns:
            Dictionary mapping IQ file paths to feature recording_session_id
        """
        from sqlalchemy import text
        
        with self.db_manager.get_session() as session:
            # Use dataset_id column directly, not extraction_metadata JSON
            # The dataset_id column is the proper foreign key reference
            query = text("""
                SELECT 
                    recording_session_id,
                    extraction_metadata->>'iq_data_path' as iq_data_path
                FROM heimdall.measurement_features
                WHERE dataset_id = :dataset_id
                  AND extraction_metadata->>'iq_data_path' IS NOT NULL
            """)
            
            result = session.execute(query, {"dataset_id": str(dataset_id)})
            
            # Build map: iq_path -> recording_session_id
            features_map = {}
            for row in result:
                recording_id = str(row[0])
                iq_path = row[1]
                
                # Normalize path (remove s3:// prefix and bucket name if present)
                iq_path = self._normalize_s3_path(iq_path)
                
                features_map[iq_path] = recording_id
            
            return features_map
    
    def _get_dataset_iq_files(self, dataset_id: UUID) -> set[str]:
        """
        Get all IQ files for a dataset from MinIO.
        
        Returns:
            Set of normalized IQ file paths
        """
        iq_files = set()
        
        # Try both possible prefix patterns
        prefixes = [
            f"synthetic/{dataset_id}/",
            f"synthetic/dataset-{dataset_id}/",
        ]
        
        for prefix in prefixes:
            try:
                paginator = self.minio_client.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.minio_client.bucket_name,
                    Prefix=prefix
                )
                
                for page in pages:
                    if "Contents" not in page:
                        continue
                    
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Only include .npy files (IQ data), skip metadata files
                        if key.endswith(".npy"):
                            iq_files.add(key)
                            
            except Exception as e:
                logger.warning(f"Error listing objects with prefix {prefix}: {e}")
        
        return iq_files
    
    def calculate_storage_size(self, dataset_id: UUID) -> int:
        """
        Calculate total storage size in bytes for a dataset from MinIO.
        
        Args:
            dataset_id: UUID of the dataset
            
        Returns:
            Total size in bytes of all files in the dataset
        """
        total_size = 0
        
        # Try both possible prefix patterns
        prefixes = [
            f"synthetic/{dataset_id}/",
            f"synthetic/dataset-{dataset_id}/",
        ]
        
        for prefix in prefixes:
            try:
                paginator = self.minio_client.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.minio_client.bucket_name,
                    Prefix=prefix
                )
                
                for page in pages:
                    if "Contents" not in page:
                        continue
                    
                    for obj in page["Contents"]:
                        # Include all files (.npy and .json)
                        total_size += obj.get("Size", 0)
                        
            except Exception as e:
                logger.warning(f"Error calculating storage size for prefix {prefix}: {e}")
        
        logger.info(f"Dataset {dataset_id} storage size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")
        return total_size
    
    def _normalize_s3_path(self, path: str) -> str:
        """
        Normalize S3 path to just the key (remove s3:// prefix and bucket name).
        
        Args:
            path: S3 path (may include s3://, bucket name, etc.)
            
        Returns:
            Normalized path (just the key)
        """
        if not path:
            return ""
        
        # Remove s3:// prefix
        if path.startswith("s3://"):
            path = path.replace("s3://", "")
            
            # Remove bucket name if present
            parts = path.split("/", 1)
            if len(parts) > 1:
                path = parts[1]
            else:
                path = parts[0]
        
        # Remove leading slash
        path = path.lstrip("/")
        
        return path
    
    def repair_dataset(
        self,
        dataset_id: UUID,
        strategy: str = "delete_orphans",
        cached_issues: Optional[list[dict]] = None
    ) -> dict:
        """
        Repair a dataset by fixing integrity issues.
        
        Strategies:
        - "delete_orphans": Delete orphaned IQ files and feature records (default)
        - "delete_iq": Only delete orphaned IQ files
        - "delete_features": Only delete orphaned feature records
        
        Args:
            dataset_id: UUID of the dataset to repair
            strategy: Repair strategy to use
            cached_issues: Optional pre-computed issues from database (avoids expensive re-validation)
            
        Returns:
            Dictionary with repair results
        """
        logger.info(f"Starting repair for dataset {dataset_id} with strategy '{strategy}'")
        
        # Get dataset type to determine which table to delete from
        dataset_type = self._get_dataset_type(dataset_id)
        
        # Use cached issues if provided, otherwise validate to get current state
        if cached_issues is not None:
            logger.info(f"Using cached validation issues ({len(cached_issues)} issues)")
            # Convert dict issues back to ValidationIssue objects
            issues = [
                ValidationIssue(
                    issue_type=issue["issue_type"],
                    path=issue["path"],
                    details=issue.get("details")
                )
                for issue in cached_issues
            ]
            # Create a minimal report-like structure
            report = type('obj', (object,), {
                'issues': issues,
                'is_healthy': len(issues) == 0
            })()
        else:
            logger.info("No cached issues provided, running full validation")
            report = self.validate_dataset(dataset_id)
        
        if report.is_healthy:
            return {
                "status": "no_action_needed",
                "message": "Dataset is already healthy",
                "deleted_iq_files": 0,
                "deleted_features": 0,
            }
        
        deleted_iq_count = 0
        deleted_features_count = 0
        
        # Delete orphaned IQ files
        if strategy in ["delete_orphans", "delete_iq"]:
            orphaned_iq_issues = [
                issue for issue in report.issues
                if issue.issue_type == "orphaned_iq"
            ]
            
            for issue in orphaned_iq_issues:
                try:
                    success, _ = self.minio_client.delete_object(issue.path)
                    if success:
                        deleted_iq_count += 1
                        logger.debug(f"Deleted orphaned IQ file: {issue.path}")
                except Exception as e:
                    logger.error(f"Failed to delete IQ file {issue.path}: {e}")
        
        # Delete orphaned feature records (batch operation)
        # Use different logic based on dataset type
        if strategy in ["delete_orphans", "delete_features"]:
            orphaned_feature_issues = [
                issue for issue in report.issues
                if issue.issue_type == "orphaned_feature"
            ]
            
            if orphaned_feature_issues:
                logger.info(f"Found {len(orphaned_feature_issues)} orphaned feature issues to process")
                
                if dataset_type == 'iq_raw':
                    # For iq_raw: Delete from synthetic_iq_samples table
                    deleted_features_count = self._repair_iq_raw_orphans(
                        dataset_id, 
                        orphaned_feature_issues
                    )
                else:
                    # For feature_based: Delete from measurement_features table
                    deleted_features_count = self._repair_feature_based_orphans(
                        dataset_id,
                        orphaned_feature_issues
                    )
                
                logger.info(f"Batch deleted {deleted_features_count} orphaned features")
            else:
                logger.info("No orphaned feature issues to repair")
        
        logger.info(
            f"Repair complete for dataset {dataset_id}: "
            f"deleted {deleted_iq_count} IQ files, {deleted_features_count} features"
        )
        
        return {
            "status": "repaired",
            "message": f"Deleted {deleted_iq_count} orphaned IQ files and {deleted_features_count} orphaned features",
            "deleted_iq_files": deleted_iq_count,
            "deleted_features": deleted_features_count,
        }
    
    def _delete_feature_record(self, recording_session_id: str) -> None:
        """Delete a single feature record from database."""
        from sqlalchemy import text
        
        with self.db_manager.get_session() as session:
            query = text("""
                DELETE FROM heimdall.measurement_features
                WHERE recording_session_id = :recording_id
            """)
            
            session.execute(query, {"recording_id": recording_session_id})
            session.commit()
    
    def _repair_iq_raw_orphans(
        self,
        dataset_id: UUID,
        orphaned_issues: list[ValidationIssue]
    ) -> int:
        """
        Repair orphaned features for iq_raw datasets.
        
        For iq_raw datasets:
        - The "features" are samples in synthetic_iq_samples table
        - Each sample has multiple IQ files (one per receiver) stored in iq_storage_paths JSONB
        - Orphaned feature = sample record exists but IQ files are missing from MinIO
        - Solution: Delete the sample record from synthetic_iq_samples
        
        Args:
            dataset_id: UUID of the dataset
            orphaned_issues: List of ValidationIssue objects with orphaned_feature type
            
        Returns:
            Number of samples deleted
        """
        if not orphaned_issues:
            logger.info("No orphaned iq_raw samples to delete")
            return 0
        
        # Extract sample_ids from issue details
        # Format: "Feature record <sample_id> exists but IQ file missing in MinIO"
        import re
        sample_ids = []
        for issue in orphaned_issues:
            if issue.details and "Feature record" in issue.details:
                match = re.search(r'Feature record ([a-f0-9\-]+)', issue.details)
                if match:
                    sample_ids.append(match.group(1))
        
        logger.info(f"Extracted {len(sample_ids)} sample_ids from {len(orphaned_issues)} issues")
        
        if not sample_ids:
            logger.warning("No sample_ids extracted from orphaned feature issues")
            return 0
        
        # Convert to UUID objects
        from uuid import UUID
        uuid_objects = [UUID(str(sid)) if not isinstance(sid, UUID) else sid 
                       for sid in sample_ids]
        
        # Batch delete from synthetic_iq_samples
        from sqlalchemy import text
        with self.db_manager.get_session() as session:
            query = text("""
                DELETE FROM heimdall.synthetic_iq_samples
                WHERE dataset_id = :dataset_id
                  AND id = ANY(:sample_ids)
            """)
            
            logger.info(f"Deleting {len(uuid_objects)} samples from synthetic_iq_samples for dataset {dataset_id}")
            result = session.execute(query, {
                "dataset_id": str(dataset_id),
                "sample_ids": uuid_objects
            })
            deleted_count = result.rowcount
            session.commit()
            
            logger.info(f"Successfully deleted {deleted_count} samples from synthetic_iq_samples")
            return deleted_count
    
    def _repair_feature_based_orphans(
        self,
        dataset_id: UUID,
        orphaned_issues: list[ValidationIssue]
    ) -> int:
        """
        Repair orphaned features for feature_based datasets.
        
        For feature_based datasets:
        - The "features" are measurement_features records
        - Each feature has a single IQ file referenced in extraction_metadata->>'iq_data_path'
        - Orphaned feature = feature record exists but IQ file is missing from MinIO
        - Solution: Delete the feature record from measurement_features
        
        Args:
            dataset_id: UUID of the dataset
            orphaned_issues: List of ValidationIssue objects with orphaned_feature type
            
        Returns:
            Number of features deleted
        """
        if not orphaned_issues:
            logger.info("No orphaned feature_based features to delete")
            return 0
        
        # Extract recording_session_ids from issue details
        # Format: "Feature record <recording_session_id> exists but IQ file missing in MinIO"
        import re
        recording_ids = []
        for issue in orphaned_issues:
            if issue.details and "Feature record" in issue.details:
                match = re.search(r'Feature record ([a-f0-9\-]+)', issue.details)
                if match:
                    recording_ids.append(match.group(1))
        
        logger.info(f"Extracted {len(recording_ids)} recording_session_ids from {len(orphaned_issues)} issues")
        
        if not recording_ids:
            logger.warning("No recording_session_ids extracted from orphaned feature issues")
            return 0
        
        # Convert to UUID objects
        from uuid import UUID
        uuid_objects = [UUID(str(rid)) if not isinstance(rid, UUID) else rid 
                       for rid in recording_ids]
        
        # Batch delete from measurement_features
        from sqlalchemy import text
        with self.db_manager.get_session() as session:
            query = text("""
                DELETE FROM heimdall.measurement_features
                WHERE dataset_id = :dataset_id
                  AND recording_session_id = ANY(:recording_ids)
            """)
            
            logger.info(f"Deleting {len(uuid_objects)} features from measurement_features for dataset {dataset_id}")
            result = session.execute(query, {
                "dataset_id": str(dataset_id),
                "recording_ids": uuid_objects
            })
            deleted_count = result.rowcount
            session.commit()
            
            logger.info(f"Successfully deleted {deleted_count} features from measurement_features")
            return deleted_count
