#!/usr/bin/env python3
"""
Test script for weighted audio sampling.

Tests that AudioLibraryLoader respects category weights from Redis
and produces the expected distribution of samples.
"""

import json
import os
import sys
from collections import Counter

import redis

# Add services path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services/training/src'))

from data.audio_library import AudioLibraryLoader, CategoryWeights


def test_weighted_sampling():
    """Test that weighted sampling produces expected distribution."""
    
    print("=" * 60)
    print("Testing Weighted Audio Sampling")
    print("=" * 60)
    
    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    
    # Set custom weights in Redis (simulate frontend slider configuration)
    # Example: Heavy bias towards voice (70%), some music (20%), rest (10%)
    custom_weights = CategoryWeights(
        voice=0.7,
        music=0.2,
        documentary=0.05,
        conference=0.05,
        custom=0.0
    )
    
    print(f"\n1. Setting custom category weights in Redis:")
    print(f"   voice: {custom_weights.voice} (70%)")
    print(f"   music: {custom_weights.music} (20%)")
    print(f"   documentary: {custom_weights.documentary} (5%)")
    print(f"   conference: {custom_weights.conference} (5%)")
    print(f"   custom: {custom_weights.custom} (0%)")
    
    redis_client.set(
        "audio:category:weights",
        json.dumps(custom_weights.to_dict())
    )
    
    # Create loader instance
    print(f"\n2. Creating AudioLibraryLoader...")
    loader = AudioLibraryLoader()
    
    # Load N samples and track category distribution
    num_samples = 100
    print(f"\n3. Loading {num_samples} random samples...")
    
    categories_selected = []
    
    for i in range(num_samples):
        try:
            audio_samples, sample_rate = loader.get_random_sample()
            
            # Extract category from last log (hacky but works for testing)
            # In production, you'd get this from the loader's internal state
            # For now, we'll use the statistics
            if (i + 1) % 10 == 0:
                print(f"   Loaded {i + 1}/{num_samples} samples...")
        
        except Exception as e:
            print(f"   ERROR loading sample {i + 1}: {e}")
            break
    
    # Get final statistics
    print(f"\n4. Analyzing category distribution...")
    stats = loader.get_stats()
    
    print(f"\n   Total chunks loaded: {stats['chunks_loaded']}")
    print(f"\n   Category distribution:")
    
    category_dist = stats.get('category_distribution', {})
    
    if category_dist:
        for cat, data in sorted(category_dist.items(), key=lambda x: x[1]['count'], reverse=True):
            count = data['count']
            percentage = data['percentage']
            print(f"     {cat:12s}: {count:3d} samples ({percentage:5.1f}%)")
    else:
        print("     No category distribution data available")
    
    # Compare with expected weights
    print(f"\n5. Comparison with expected weights:")
    print(f"   {'Category':<15} {'Expected':<10} {'Actual':<10} {'Difference':<10}")
    print(f"   {'-' * 50}")
    
    expected_weights = custom_weights.normalize().to_dict()
    
    for cat, expected_pct in expected_weights.items():
        if expected_pct == 0:
            continue
        
        actual_data = category_dist.get(cat, {'percentage': 0.0})
        actual_pct = actual_data['percentage'] / 100.0  # Convert to decimal
        difference = actual_pct - expected_pct
        
        status = "✓" if abs(difference) < 0.15 else "⚠"  # Within 15% tolerance
        
        print(f"   {cat:<15} {expected_pct * 100:>6.1f}%    {actual_pct * 100:>6.1f}%    {difference * 100:>+6.1f}%  {status}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    # Verify the distribution is reasonable
    if category_dist:
        voice_pct = category_dist.get('voice', {}).get('percentage', 0.0) / 100.0
        music_pct = category_dist.get('music', {}).get('percentage', 0.0) / 100.0
        
        if voice_pct > music_pct and voice_pct > 0.5:
            print("\n✅ SUCCESS: Weighted sampling is working correctly!")
            print(f"   Voice dominates as expected ({voice_pct * 100:.1f}%)")
            return True
        else:
            print("\n⚠️  WARNING: Distribution doesn't match expected weights")
            print(f"   Voice: {voice_pct * 100:.1f}%, Music: {music_pct * 100:.1f}%")
            return False
    else:
        print("\n❌ FAILED: No samples loaded or no category distribution")
        return False


if __name__ == "__main__":
    success = test_weighted_sampling()
    sys.exit(0 if success else 1)
