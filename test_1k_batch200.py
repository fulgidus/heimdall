#!/usr/bin/env python3
"""
Test GPU batch processing with 1000 samples using batch_size=200

This script validates the optimized batch processing architecture with production-scale datasets.
Expected throughput: 8-10 samples/sec based on 200-sample benchmarks.
"""

import requests
import json
import time
import sys


def test_1k_samples_batch200():
    """Test 1000-sample dataset generation with batch_size=200"""
    
    print("=" * 80)
    print("GPU Batch Processing Test - 1000 Samples (batch_size=200)")
    print("=" * 80)
    print()
    
    base_url = "http://localhost:8002"
    
    # Test configuration
    num_samples = 1000
    expected_batch_size = 200  # Should be configured in synthetic_generator.py
    
    payload = {
        "name": "benchmark_1k_batch200",
        "description": "Benchmark test for 1000 samples with batch_size=200",
        "num_samples": num_samples,
        "frequency_mhz": 145.0,
        "tx_power_dbm": 50.0,  # Higher power for better SNR
        "min_snr_db": 5.0,  # Lower threshold to accept more samples
        "min_receivers": 3,
        "max_gdop": 100.0,  # Maximum permissive GDOP
        "dataset_type": "feature_based",  # Use feature_based to enable GPU processing
        "use_random_receivers": True,
        "seed": 43  # Different seed for new test
    }
    
    print(f"Configuration:")
    print(f"  - Samples: {num_samples}")
    print(f"  - Expected batch size: {expected_batch_size}")
    print(f"  - Expected batches: {num_samples // expected_batch_size}")
    print(f"  - Dataset type: features")
    print(f"  - Frequency: 145.0 MHz")
    print()
    
    # Start generation
    print(f"Starting dataset generation...")
    start_time = time.time()
    
    try:
        # Submit job
        response = requests.post(
            f"{base_url}/v1/training/synthetic/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        job_info = response.json()
        job_id = job_info["job_id"]
        
        print(f"Job submitted: {job_id}")
        print(f"Polling for completion...")
        
        # Poll for completion
        while True:
            time.sleep(5)  # Poll every 5 seconds
            status_response = requests.get(
                f"{base_url}/v1/training/synthetic/jobs/{job_id}",
                timeout=30
            )
            status_response.raise_for_status()
            status_info = status_response.json()
            
            status = status_info["status"]
            progress = status_info.get("current_progress", 0)
            total = status_info.get("total_progress", num_samples)
            
            print(f"  Status: {status} - Progress: {progress}/{total} ({(progress/total*100):.1f}%)")
            
            if status in ["completed", "failed", "cancelled"]:
                break
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if status != "completed":
            print(f"\n❌ Job failed with status: {status}")
            if "error" in status_info:
                print(f"   Error: {status_info['error']}")
            return False
        
        result = status_info
        
        # Calculate metrics
        throughput = num_samples / elapsed
        time_per_sample = elapsed / num_samples
        expected_batches = num_samples // expected_batch_size
        time_per_batch = elapsed / expected_batches
        
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"✅ Success!")
        print()
        print(f"Dataset Statistics:")
        print(f"  - Samples generated: {result.get('current_progress', 'N/A')}")
        print(f"  - Dataset ID: {result.get('dataset_id', 'N/A')}")
        print()
        print(f"Performance Metrics:")
        print(f"  - Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"  - Throughput: {throughput:.2f} samples/sec")
        print(f"  - Time per sample: {time_per_sample*1000:.1f} ms")
        print(f"  - Estimated batches: {expected_batches}")
        print(f"  - Time per batch: {time_per_batch:.2f} seconds")
        print()
        
        # Compare to baseline (from session summary: 2.49 samples/sec for 10 samples)
        baseline_throughput = 2.49
        speedup = throughput / baseline_throughput
        
        print(f"Comparison to Baseline (10 samples):")
        print(f"  - Baseline throughput: {baseline_throughput:.2f} samples/sec")
        print(f"  - Current throughput: {throughput:.2f} samples/sec")
        print(f"  - Speedup: {speedup:.2f}x")
        print()
        
        # Compare to 200-sample benchmark (from session summary: 9.00 samples/sec)
        benchmark_200_throughput = 9.00
        efficiency = (throughput / benchmark_200_throughput) * 100
        
        print(f"Comparison to 200-Sample Benchmark:")
        print(f"  - Benchmark throughput: {benchmark_200_throughput:.2f} samples/sec")
        print(f"  - Current throughput: {throughput:.2f} samples/sec")
        print(f"  - Efficiency: {efficiency:.1f}%")
        print()
        
        # Estimate GPU initialization overhead
        # Assuming ~4.5s GPU init per batch + ~15ms per chunk
        samples_per_batch = expected_batch_size
        chunks_per_sample = 5  # From session notes
        chunks_per_batch = samples_per_batch * chunks_per_sample
        estimated_gpu_init = 4.5 * expected_batches
        estimated_extraction = 0.015 * chunks_per_batch * expected_batches  # 15ms per chunk
        estimated_total = estimated_gpu_init + estimated_extraction
        
        print(f"GPU Overhead Analysis:")
        print(f"  - Estimated GPU init overhead: {estimated_gpu_init:.1f}s ({(estimated_gpu_init/elapsed)*100:.1f}% of total)")
        print(f"  - Estimated extraction time: {estimated_extraction:.1f}s ({(estimated_extraction/elapsed)*100:.1f}% of total)")
        print(f"  - Estimated total: {estimated_total:.1f}s")
        print(f"  - Actual total: {elapsed:.1f}s")
        print(f"  - Difference (IQ gen + overhead): {(elapsed - estimated_total):.1f}s")
        print()
        
        # Results summary
        if 'results' in result:
            res = result['results']
            print(f"Generation Results:")
            print(f"  - Total generated: {res.get('num_samples', 'N/A')}")
            print(f"  - Dataset path: {res.get('dataset_path', 'N/A')}")
            print()
        
        print("=" * 80)
        
        return True
        
    except requests.exceptions.Timeout:
        print()
        print("❌ Request timed out (>10 minutes)")
        print("   Consider increasing timeout or reducing sample count")
        return False
        
    except requests.exceptions.RequestException as e:
        print()
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Response: {e.response.text}")
        return False
        
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("🚀 Starting 1000-sample batch_size=200 benchmark")
    print()
    
    success = test_1k_samples_batch200()
    
    sys.exit(0 if success else 1)
