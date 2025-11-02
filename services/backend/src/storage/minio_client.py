"""MinIO S3 storage client for IQ data."""

import json
import logging
from io import BytesIO

import boto3
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class MinIOClient:
    """Client for storing IQ data and metadata in MinIO S3."""

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "heimdall-raw-iq",
        region_name: str = "us-east-1",
    ):
        """Initialize MinIO S3 client."""
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.region_name = region_name

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )

        logger.info(
            "MinIO client initialized - endpoint: %s, bucket: %s", endpoint_url, bucket_name
        )

    def ensure_bucket_exists(self) -> bool:
        """Ensure the bucket exists, create if not."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug("Bucket %s already exists", self.bucket_name)
            return True
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info("Created bucket: %s", self.bucket_name)
                    return True
                except ClientError as create_error:
                    logger.error("Failed to create bucket %s: %s", self.bucket_name, create_error)
                    return False
            else:
                logger.error("Error checking bucket %s: %s", self.bucket_name, e)
                return False
        except NoCredentialsError:
            logger.error("MinIO credentials not found")
            return False

    def upload_iq_data(
        self,
        iq_data: np.ndarray,
        task_id: str,
        websdr_id: int,
        metadata: dict | None = None,
    ) -> tuple[bool, str]:
        """Upload IQ data as .npy file to MinIO."""
        try:
            if not self.ensure_bucket_exists():
                return False, f"Failed to access bucket {self.bucket_name}"

            s3_path = f"sessions/{task_id}/websdr_{websdr_id}.npy"

            buffer = BytesIO()
            np.save(buffer, iq_data)
            buffer.seek(0)
            iq_bytes = buffer.getvalue()

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=iq_bytes,
                ContentType="application/octet-stream",
                Metadata={
                    "websdr_id": str(websdr_id),
                    "task_id": task_id,
                    "data_type": "iq_complex64",
                    "samples_count": str(len(iq_data)),
                },
            )

            logger.info(
                "Uploaded IQ data to s3://%s/%s (%d samples, %.2f MB)",
                self.bucket_name,
                s3_path,
                len(iq_data),
                iq_bytes.__sizeof__() / (1024 * 1024),
            )

            if metadata:
                metadata_path = f"sessions/{task_id}/websdr_{websdr_id}_metadata.json"
                metadata_json = json.dumps(metadata, indent=2)

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_path,
                    Body=metadata_json.encode("utf-8"),
                    ContentType="application/json",
                    Metadata={
                        "websdr_id": str(websdr_id),
                        "task_id": task_id,
                        "data_type": "metadata",
                    },
                )

                logger.info("Uploaded metadata to s3://%s/%s", self.bucket_name, metadata_path)

            return True, f"s3://{self.bucket_name}/{s3_path}"

        except ClientError as e:
            error_msg = f"Failed to upload IQ data: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error uploading IQ data: {str(e)}"
            logger.exception(error_msg)
            return False, error_msg

    def download_iq_data(
        self,
        task_id: str = None,
        websdr_id: int = None,
        s3_path: str = None,
    ) -> bytes:
        """
        Download IQ data from MinIO.
        
        Args:
            task_id: Task ID (legacy format)
            websdr_id: WebSDR ID (legacy format)
            s3_path: Direct S3 path (new format for real recordings)
            
        Returns:
            Raw IQ data bytes
            
        Raises:
            Exception if download fails
        """
        try:
            if s3_path:
                # Direct path provided (for real recordings)
                # Remove bucket prefix if present
                if s3_path.startswith(f"s3://{self.bucket_name}/"):
                    s3_path = s3_path.replace(f"s3://{self.bucket_name}/", "")
                elif s3_path.startswith("s3://"):
                    # Extract path after bucket name
                    parts = s3_path.replace("s3://", "").split("/", 1)
                    s3_path = parts[1] if len(parts) > 1 else parts[0]
                    
                key = s3_path
            else:
                # Legacy format (task_id + websdr_id)
                key = f"sessions/{task_id}/websdr_{websdr_id}.npy"

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            iq_bytes = response["Body"].read()

            logger.info(
                "Downloaded IQ data from s3://%s/%s (%d bytes)",
                self.bucket_name,
                key,
                len(iq_bytes),
            )

            return iq_bytes

        except ClientError as e:
            logger.error("Failed to download IQ data from %s: %s", key if 'key' in locals() else 'unknown', e)
            raise
        except Exception as e:
            logger.exception("Unexpected error downloading IQ data: %s", e)
            raise

    def get_session_measurements(self, task_id: str) -> dict[int, dict]:
        """List all measurements from a session."""
        measurements = {}

        try:
            prefix = f"sessions/{task_id}/"

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]

                    if ".npy" in key and "_metadata" not in key:
                        parts = key.split("/")
                        if parts[-1].startswith("websdr_"):
                            websdr_id = int(parts[-1].replace("websdr_", "").replace(".npy", ""))
                            measurements[websdr_id] = {
                                "iq_data_path": f"s3://{self.bucket_name}/{key}",
                                "size_bytes": obj["Size"],
                                "last_modified": obj["LastModified"].isoformat(),
                            }

            logger.info("Found %d measurements in session %s", len(measurements), task_id)

            return measurements

        except ClientError as e:
            logger.error("Failed to list session measurements: %s", e)
            return {}
        except Exception as e:
            logger.exception("Unexpected error listing measurements: %s", e)
            return {}

    def delete_object(self, s3_path: str) -> tuple[bool, str]:
        """Delete a single object from MinIO."""
        try:
            # Remove s3:// prefix and bucket name if present
            if s3_path.startswith("s3://"):
                s3_path = s3_path.replace(f"s3://{self.bucket_name}/", "")

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_path)
            logger.info("Deleted object: s3://%s/%s", self.bucket_name, s3_path)
            return True, f"Deleted s3://{self.bucket_name}/{s3_path}"

        except ClientError as e:
            error_msg = f"Failed to delete object {s3_path}: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error deleting object {s3_path}: {str(e)}"
            logger.exception(error_msg)
            return False, error_msg

    def delete_session_data(self, session_id: str) -> tuple[int, int]:
        """Delete all IQ data files for a session.
        
        Returns:
            tuple[int, int]: (successful_deletes, failed_deletes)
        """
        successful = 0
        failed = 0

        try:
            prefix = f"sessions/{session_id}/"

            # List all objects with this prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    objects_to_delete.append({"Key": obj["Key"]})

            if not objects_to_delete:
                logger.info("No objects found for session %s", session_id)
                return 0, 0

            # Delete objects in batches (max 1000 per request)
            batch_size = 1000
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i : i + batch_size]

                response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name, Delete={"Objects": batch}
                )

                if "Deleted" in response:
                    successful += len(response["Deleted"])

                if "Errors" in response:
                    failed += len(response["Errors"])
                    for error in response["Errors"]:
                        logger.error(
                            "Failed to delete %s: %s", error["Key"], error["Message"]
                        )

            logger.info(
                "Deleted session %s data: %d successful, %d failed",
                session_id,
                successful,
                failed,
            )

            return successful, failed

        except ClientError as e:
            logger.error("Failed to delete session data for %s: %s", session_id, e)
            return successful, failed
        except Exception as e:
            logger.exception("Unexpected error deleting session data for %s: %s", session_id, e)
            return successful, failed

    def health_check(self) -> dict[str, bool]:
        """Check MinIO connectivity and bucket access."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)

            return {
                "status": "healthy",
                "endpoint": self.endpoint_url,
                "bucket": self.bucket_name,
                "accessible": True,
            }
        except Exception as e:
            logger.error("MinIO health check failed: %s", e)
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint_url,
                "bucket": self.bucket_name,
                "accessible": False,
                "error": str(e),
            }


import numpy as np

logger = logging.getLogger(__name__)
