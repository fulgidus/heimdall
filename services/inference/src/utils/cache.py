"""Redis cache manager for Phase 6 Inference Service.

Implements caching strategy for prediction results to achieve >80% cache hit rate
and reduce latency for repeated queries.
"""

import redis
import json
import logging
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based cache for inference predictions.
    
    Caching strategy:
    - Key: hash(preprocessed_features) - stable and deterministic
    - Value: JSON-serialized prediction result
    - TTL: Configurable (default 3600 seconds = 1 hour)
    - Hit rate target: >80%
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 3600,
        password: Optional[str] = None,
    ):
        """
        Initialize Redis cache connection.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            ttl_seconds: Cache entry TTL in seconds (default 1 hour)
            password: Redis password (optional)
        """
        self.ttl_seconds = ttl_seconds
        self.host = host
        self.port = port
        self.db = db
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,  # Return strings instead of bytes
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_cache_key(self, features: np.ndarray) -> str:
        """
        Generate stable cache key from features.
        
        Args:
            features: Preprocessed features (mel-spectrogram)
        
        Returns:
            Cache key string
        """
        try:
            # Convert to bytes for hashing
            features_bytes = features.astype(np.float32).tobytes()
            
            # Create SHA256 hash
            hash_obj = hashlib.sha256(features_bytes)
            cache_key = f"pred:{hash_obj.hexdigest()}"
            
            logger.debug(f"Generated cache key: {cache_key}")
            return cache_key
        
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise
    
    def get(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction if available.
        
        Args:
            features: Preprocessed features
        
        Returns:
            Cached prediction dict, or None if not found
        """
        try:
            cache_key = self._generate_cache_key(features)
            
            # Retrieve from Redis
            cached_value = self.client.get(cache_key)
            
            if cached_value is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
            
            # Deserialize JSON
            result = json.loads(cached_value)
            result['_cache_hit'] = True
            
            logger.debug(f"Cache hit: {cache_key}")
            return result
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to deserialize cached value: {e}")
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, features: np.ndarray, prediction: Dict[str, Any]) -> bool:
        """
        Cache prediction result.
        
        Args:
            features: Preprocessed features
            prediction: Prediction result dict
        
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(features)
            
            # Prepare value for storage (make it JSON-serializable)
            cache_value = self._prepare_for_cache(prediction)
            
            # Serialize to JSON
            json_value = json.dumps(cache_value)
            
            # Store in Redis with TTL
            success = self.client.setex(
                cache_key,
                self.ttl_seconds,
                json_value
            )
            
            if success:
                logger.debug(f"Cached prediction: {cache_key} (TTL: {self.ttl_seconds}s)")
            
            return success
        
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def _prepare_for_cache(self, obj: Any) -> Any:
        """
        Prepare object for JSON serialization.
        
        Converts numpy types to native Python types.
        
        Args:
            obj: Object to prepare
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_cache(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_cache(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def delete(self, features: np.ndarray) -> bool:
        """
        Delete cached prediction.
        
        Args:
            features: Preprocessed features
        
        Returns:
            True if deleted, False if not found
        """
        try:
            cache_key = self._generate_cache_key(features)
            deleted = self.client.delete(cache_key)
            logger.debug(f"Deleted cache entry: {cache_key}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries in current database.
        
        WARNING: This clears the entire Redis database!
        
        Returns:
            True if successful
        """
        try:
            self.client.flushdb()
            logger.warning(f"Cleared all cache entries in database {self.db}")
            return True
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from Redis.
        
        Returns:
            Statistics dict with cache info
        """
        try:
            info = self.client.info(section='memory')
            
            stats = {
                'used_memory_bytes': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'total_keys': self.client.dbsize(),
                'connection_host': self.host,
                'connection_port': self.port,
                'connection_db': self.db,
                'ttl_seconds': self.ttl_seconds,
            }
            
            logger.debug(f"Cache stats: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def close(self):
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")


class CacheStatistics:
    """Track cache hit/miss statistics."""
    
    def __init__(self):
        """Initialize statistics."""
        self.hits = 0
        self.misses = 0
    
    @property
    def total(self) -> int:
        """Total accesses."""
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total
    
    def record_hit(self):
        """Record cache hit."""
        self.hits += 1
    
    def record_miss(self):
        """Record cache miss."""
        self.misses += 1
    
    def reset(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"total={self.total}, hit_rate={self.hit_rate:.1%})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Return as dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': self.total,
            'hit_rate': self.hit_rate,
        }


def create_cache(
    host: str = "localhost",
    port: int = 6379,
    ttl_seconds: int = 3600,
) -> Optional[RedisCache]:
    """
    Factory function to create Redis cache with error handling.
    
    Args:
        host: Redis host
        port: Redis port
        ttl_seconds: Cache TTL
    
    Returns:
        RedisCache instance, or None if connection failed
    """
    try:
        return RedisCache(host=host, port=port, ttl_seconds=ttl_seconds)
    except Exception as e:
        logger.error(f"Failed to create cache: {e}")
        return None
