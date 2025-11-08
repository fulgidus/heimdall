"""
End-to-End tests for OpenWebRX acquisition.

These tests verify the complete RF acquisition workflow:
1. Submit acquisition task via API
2. Celery worker fetches IQ data from OpenWebRX receivers
3. Data is processed and stored in MinIO
4. Metadata is saved to TimescaleDB
5. Task completes successfully with correct measurements
"""

import asyncio
from typing import Any

import pytest
from httpx import AsyncClient


class TestOpenWebRXAcquisition:
    """E2E tests for OpenWebRX RF acquisition workflow."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_acquisition_single_frequency(self, api_client: AsyncClient):
        """
        Test basic acquisition at a single frequency.

        Steps:
        1. Submit acquisition task for 145.5 MHz
        2. Wait for task completion (max 30 seconds)
        3. Verify task succeeded
        4. Verify measurements collected from all 7 WebSDRs
        5. Verify metadata format
        """
        # Step 1: Submit acquisition task
        response = await api_client.post(
            "/api/v1/acquisition/acquire",
            json={
                "frequency_mhz": 145.5,
                "duration_seconds": 3.0,
                "sample_rate_khz": 12.0,
            },
        )

        assert response.status_code == 200, f"Failed to submit task: {response.text}"

        data = response.json()
        assert "task_id" in data
        assert data["status"] == "PENDING"
        assert data["frequency_mhz"] == 145.5
        assert data["websdrs_count"] == 7

        task_id = data["task_id"]
        print(f"\n✅ Task submitted: {task_id}")

        # Step 2: Poll for completion
        max_wait = 30  # seconds
        poll_interval = 2  # seconds
        elapsed = 0

        task_status = None

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            response = await api_client.get(f"/api/v1/acquisition/status/{task_id}")

            assert response.status_code == 200
            task_status = response.json()

            status = task_status["status"]
            progress = task_status.get("progress", 0)

            print(f"⏳ [{elapsed}s] Status: {status}, Progress: {progress:.1f}%")

            if status in ["SUCCESS", "FAILURE"]:
                break

        # Step 3: Verify task succeeded
        assert task_status is not None, "No status received"
        assert (
            task_status["status"] == "SUCCESS"
        ), f"Task failed: {task_status.get('message', 'Unknown error')}"

        print("✅ Task completed successfully")

        # Step 4: Verify measurements
        measurements_collected = task_status["measurements_collected"]
        assert measurements_collected == 7, f"Expected 7 measurements, got {measurements_collected}"

        print("✅ All 7 WebSDRs collected data")

        # Step 5: Verify metadata format
        result = task_status["result"]
        assert result is not None
        assert "measurements" in result
        assert len(result["measurements"]) == 7

        for idx, measurement in enumerate(result["measurements"], 1):
            # Check required fields
            assert "websdr_id" in measurement
            assert "frequency_mhz" in measurement
            assert "timestamp_utc" in measurement
            assert "metrics" in measurement
            assert "iq_data_path" in measurement

            # Verify metrics structure
            metrics = measurement["metrics"]
            assert "snr_db" in metrics
            assert "psd_dbm" in metrics
            assert "frequency_offset_hz" in metrics

            # Verify S3 path format
            iq_path = measurement["iq_data_path"]
            assert iq_path.startswith("s3://heimdall-raw-iq/sessions/")
            assert iq_path.endswith(f"websdr_{measurement['websdr_id']}.npy")

            print(
                f"  ✅ WebSDR {idx}: SNR={metrics['snr_db']:.2f} dB, "
                f"PSD={metrics['psd_dbm']:.2f} dBm, "
                f"Samples={measurement['samples_count']}"
            )

        print("\n🎉 E2E test PASSED: All 7 WebSDRs working correctly!")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_acquisition_multiple_frequencies(self, api_client: AsyncClient):
        """
        Test acquisition at multiple frequencies sequentially.

        This verifies that the system can handle multiple acquisitions
        in sequence without interference.
        """
        frequencies = [144.8, 145.5, 432.1]
        results = []

        for freq in frequencies:
            print(f"\n📡 Testing frequency: {freq} MHz")

            # Submit task
            response = await api_client.post(
                "/api/v1/acquisition/acquire",
                json={
                    "frequency_mhz": freq,
                    "duration_seconds": 2.0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            task_id = data["task_id"]

            # Wait for completion
            max_wait = 30
            poll_interval = 2
            elapsed = 0
            task_status = None

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                response = await api_client.get(f"/api/v1/acquisition/status/{task_id}")

                task_status = response.json()

                if task_status["status"] in ["SUCCESS", "FAILURE"]:
                    break

            assert task_status is not None, "No status received"
            assert task_status["status"] == "SUCCESS"
            results.append(
                {"frequency": freq, "measurements": task_status["measurements_collected"]}
            )

            print(f"  ✅ {freq} MHz: {task_status['measurements_collected']} measurements")

        # Verify all frequencies worked (allow at least 5/7 WebSDRs)
        assert len(results) == len(frequencies)
        for result in results:
            assert (
                result["measurements"] >= 5
            ), f"Expected at least 5 measurements, got {result['measurements']}"

        print("\n🎉 Multi-frequency test PASSED! (≥5/7 WebSDRs responded)")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_acquisition_error_handling(self, api_client: AsyncClient):
        """
        Test error handling for invalid parameters.
        """
        # Test invalid frequency (too low)
        response = await api_client.post(
            "/api/v1/acquisition/acquire",
            json={
                "frequency_mhz": 10.0,  # Below 2m band
                "duration_seconds": 3.0,
            },
        )

        # Should still accept the task (validation happens in worker)
        # or return validation error
        assert response.status_code in [200, 422]

        # Test invalid duration (negative)
        response = await api_client.post(
            "/api/v1/acquisition/acquire",
            json={
                "frequency_mhz": 145.5,
                "duration_seconds": -1.0,
            },
        )

        assert response.status_code == 422  # Validation error

        # Test missing required fields
        response = await api_client.post("/api/v1/acquisition/acquire", json={})

        assert response.status_code == 422

        print("✅ Error handling tests PASSED")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_acquisition_metadata_correlation(self, api_client: AsyncClient):
        """
        Test that acquisition includes proper correlation metadata.

        This verifies that when we know the source (e.g., a test beacon),
        we can properly tag the acquisition with source metadata.
        """
        # Submit acquisition with metadata
        response = await api_client.post(
            "/api/v1/acquisition/acquire",
            json={
                "frequency_mhz": 145.5,
                "duration_seconds": 3.0,
                # TODO: Add source metadata once implemented
                # "metadata": {
                #     "source_callsign": "IK1JNS",
                #     "source_type": "beacon",
                #     "notes": "Test acquisition"
                # }
            },
        )

        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]

        # Wait for completion
        max_wait = 30
        elapsed = 0
        task_status: dict[str, Any] = {}

        while elapsed < max_wait:
            await asyncio.sleep(2)
            elapsed += 2

            response = await api_client.get(f"/api/v1/acquisition/status/{task_id}")

            task_status = response.json()

            if task_status["status"] in ["SUCCESS", "FAILURE"]:
                break

        assert task_status.get("status") is not None, "No status received"
        assert task_status["status"] == "SUCCESS"

        # TODO: Verify metadata once implemented
        # result = task_status["result"]
        # assert "metadata" in result
        # assert result["metadata"]["source_callsign"] == "IK1JNS"

        print("✅ Metadata correlation test PASSED (partial - TODO)")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_concurrent_acquisitions(self, api_client: AsyncClient):
        """
        Test multiple concurrent acquisitions.

        This verifies that the system can handle multiple simultaneous
        acquisition requests without interference.
        """
        num_concurrent = 3
        frequencies = [144.8, 145.5, 432.1]

        # Submit all tasks concurrently
        task_ids = []

        for freq in frequencies:
            response = await api_client.post(
                "/api/v1/acquisition/acquire",
                json={
                    "frequency_mhz": freq,
                    "duration_seconds": 2.0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            task_ids.append((freq, data["task_id"]))
            print(f"📡 Submitted {freq} MHz: {data['task_id']}")

        # Wait for all to complete
        max_wait = 45  # seconds
        poll_interval = 3
        elapsed = 0

        completed = set()

        while elapsed < max_wait and len(completed) < num_concurrent:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            for freq, task_id in task_ids:
                if task_id in completed:
                    continue

                response = await api_client.get(f"/api/v1/acquisition/status/{task_id}")

                task_status = response.json()
                status = task_status["status"]

                if status in ["SUCCESS", "FAILURE"]:
                    completed.add(task_id)
                    print(f"  ✅ {freq} MHz completed: {status}")

        # Verify all completed successfully
        assert (
            len(completed) == num_concurrent
        ), f"Only {len(completed)}/{num_concurrent} tasks completed"

        # Check individual results (allow at least 5/7 WebSDRs for concurrent)
        for freq, task_id in task_ids:
            response = await api_client.get(f"/api/v1/acquisition/status/{task_id}")

            task_status = response.json()
            assert task_status["status"] == "SUCCESS"
            assert (
                task_status["measurements_collected"] >= 5
            ), f"{freq} MHz: Expected at least 5 measurements, got {task_status['measurements_collected']}"

        print("\n🎉 Concurrent acquisitions test PASSED! (≥5/7 WebSDRs per task)")


class TestOpenWebRXWebSocketProtocol:
    """Direct WebSocket protocol tests (without full API)."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_websocket_handshake(self):
        """
        Test direct WebSocket connection and handshake.

        This verifies the OpenWebRX protocol implementation at a low level.
        """
        import websockets

        url = "ws://sdr1.ik1jns.it:8076/ws/"

        async with websockets.connect(url, max_size=10_000_000) as ws:
            # Send handshake
            await ws.send("SERVER DE CLIENT client=heimdall type=receiver")

            # Wait for server acknowledgment
            timeout_count = 0
            handshake_received = False

            for _ in range(10):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)

                    if isinstance(msg, str) and "CLIENT DE SERVER" in msg:
                        handshake_received = True
                        print(f"✅ Handshake received: {msg}")
                        break

                except TimeoutError:
                    timeout_count += 1
                    if timeout_count > 3:
                        break

            assert handshake_received, "Server handshake not received"

            # Set frequency
            await ws.send("SET mod=iq freq=145000000")
            await asyncio.sleep(0.5)

            # Start stream
            await ws.send("START")

            # Wait a bit for stream to start
            await asyncio.sleep(1.0)

            # Verify we receive binary data
            binary_received = False
            messages_received = 0

            for attempt in range(20):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    messages_received += 1

                    if isinstance(msg, bytes):
                        binary_received = True
                        print(
                            f"✅ Binary IQ data received: {len(msg)} bytes (after {messages_received} messages)"
                        )
                        break
                    else:
                        # JSON status message, continue waiting
                        pass

                except TimeoutError:
                    print(f"⏳ Attempt {attempt + 1}/20: timeout")
                    if attempt >= 5:  # Give up after 5 timeouts
                        break

            assert binary_received, f"No binary IQ data received (got {messages_received} messages)"

            # Stop stream
            await ws.send("STOP")

        print("✅ WebSocket protocol test PASSED")


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_health_check(api_client: AsyncClient):
    """Test that the service is healthy before running E2E tests."""
    response = await api_client.get("/health")
    assert response.status_code == 200

    health = response.json()
    assert health["status"] == "healthy"

    print(f"✅ Service healthy: {health}")


if __name__ == "__main__":
    """Run E2E tests standalone."""
    import sys

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "e2e"]))
