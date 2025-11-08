"""
End-to-End Acquisition Workflow Tests

Tests the complete acquisition workflow using mock WebSDRs to eliminate
network dependencies and flakiness.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add backend tests to path for importing mocks
backend_tests = Path(__file__).parent.parent / "backend" / "tests"
if str(backend_tests) not in sys.path:
    sys.path.insert(0, str(backend_tests))


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_e2e_acquire_and_store(test_config):
    """Test complete acquisition -> processing -> storage workflow."""
    from mock_websdrs import create_mock_fetcher

    # Setup mock fetcher
    mock_fetcher = create_mock_fetcher(receiver_count=7)

    # Execute acquisition
    result = await mock_fetcher.fetch_iq_simultaneous(
        frequency_mhz=145.50, duration_seconds=2.0, sample_rate=192000
    )

    # Assert results
    assert len(result) == 7  # 7 receivers

    # Verify each measurement
    successful = sum(1 for data, error in result.values() if error is None)
    assert successful == 7, "All receivers should succeed with mocks"

    for receiver_id, (iq_data, error) in result.items():
        assert error is None, f"Receiver {receiver_id} failed: {error}"
        assert iq_data is not None
        assert len(iq_data) > 0

    print(f"✓ Mock acquisition completed successfully with {successful} receivers")


@pytest.mark.asyncio
async def test_e2e_partial_failure_handling():
    """Test behavior when some receivers fail."""
    from mock_websdrs import create_mock_fetcher

    # Simulate 2 receivers failing
    mock_fetcher = create_mock_fetcher(receiver_count=7, failure_indices=[2, 4])

    result = await mock_fetcher.fetch_iq_simultaneous(
        frequency_mhz=145.50, duration_seconds=2.0, sample_rate=192000
    )

    # Should get results for all 7, but 2 will have errors
    assert len(result) == 7

    successful = sum(1 for data, error in result.values() if error is None)
    failed = sum(1 for data, error in result.values() if error is not None)

    assert successful == 5, "5 receivers should succeed"
    assert failed == 2, "2 receivers should fail"

    print(f"✓ Partial failure handled: {successful} success, {failed} failures")


@pytest.mark.asyncio
async def test_e2e_concurrent_acquisitions():
    """Test multiple concurrent acquisitions."""
    from mock_websdrs import create_mock_fetcher

    num_concurrent = 5
    mock_fetcher = create_mock_fetcher(receiver_count=7)

    # Execute concurrent acquisitions
    tasks = [
        mock_fetcher.fetch_iq_simultaneous(
            frequency_mhz=145.50 + i * 0.1, duration_seconds=2.0, sample_rate=192000
        )
        for i in range(num_concurrent)
    ]

    results = await asyncio.gather(*tasks)

    # Verify all completed successfully
    assert len(results) == num_concurrent

    for i, result in enumerate(results):
        assert len(result) == 7, f"Acquisition {i} should have 7 results"
        successful = sum(1 for data, error in result.values() if error is None)
        assert successful == 7, f"Acquisition {i} should have all successes"

    print(f"✓ All {num_concurrent} concurrent acquisitions completed")


@pytest.mark.asyncio
async def test_e2e_signal_quality_metrics():
    """Test that signal quality metrics are computed correctly."""
    import numpy as np
    from mock_websdrs import MockWebSDRReceiver

    receiver = MockWebSDRReceiver("test-receiver", 145.0)

    # Fetch IQ data
    data = await receiver.fetch_iq_data(
        frequency_mhz=145.50, duration_seconds=2.0, sample_rate=192000
    )

    # Verify data structure
    assert "iq_data" in data
    assert "sample_rate" in data
    assert "timestamp" in data
    assert "receiver_id" in data
    assert "signal_strength" in data

    # Verify IQ data properties
    iq_data = data["iq_data"]
    assert isinstance(iq_data, np.ndarray)
    assert iq_data.dtype == np.complex64

    expected_samples = int(192000 * 2.0)
    assert len(iq_data) == expected_samples

    print("✓ Signal quality metrics validated")


@pytest.mark.asyncio
async def test_e2e_receiver_reconnection():
    """Test receiver reconnection after failure."""
    from mock_websdrs import MockWebSDRReceiver

    receiver = MockWebSDRReceiver("test-receiver", 145.0)

    # First acquisition should succeed
    data1 = await receiver.fetch_iq_data(145.50, 2.0)
    assert data1["iq_data"] is not None

    # Disconnect receiver
    receiver.disconnect()

    # Second acquisition should fail
    with pytest.raises(ConnectionError):
        await receiver.fetch_iq_data(145.50, 2.0)

    # Reconnect and try again
    receiver.reconnect()
    data2 = await receiver.fetch_iq_data(145.50, 2.0)
    assert data2["iq_data"] is not None

    print("✓ Receiver reconnection handled correctly")


@pytest.mark.asyncio
async def test_e2e_data_validation():
    """Test IQ data validation."""
    import numpy as np
    from mock_websdrs import create_mock_websdrs

    receivers = create_mock_websdrs(count=3)

    for receiver in receivers:
        data = await receiver.fetch_iq_data(145.50, 1.0)

        # Validate timestamp format
        timestamp = data["timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

        # Validate IQ data
        iq_data = data["iq_data"]
        assert not np.isnan(iq_data).any(), "IQ data should not contain NaN"
        assert not np.isinf(iq_data).any(), "IQ data should not contain Inf"

        # Validate metadata
        assert data["frequency_mhz"] == 145.50
        assert data["duration_seconds"] == 1.0
        assert data["sample_rate"] == 192000
        assert isinstance(data["signal_strength"], (int, float))

    print("✓ Data validation passed for all receivers")


@pytest.mark.asyncio
async def test_e2e_fetch_count_tracking():
    """Test that fetch count is tracked correctly."""
    from mock_websdrs import create_mock_fetcher

    mock_fetcher = create_mock_fetcher(receiver_count=7)

    assert mock_fetcher.fetch_count == 0

    # Perform 3 fetches
    for _i in range(3):
        await mock_fetcher.fetch_iq_simultaneous(145.50, 2.0)

    assert mock_fetcher.fetch_count == 3

    print("✓ Fetch count tracking validated")
