"""Simple test to check module import"""


def test_import_base():
    """Test if Base can be imported"""
    from src.models.db import Base

    assert Base is not None


def test_import_measurement():
    """Test if Measurement can be imported"""
    from src.models.db import Measurement

    assert Measurement is not None
    assert hasattr(Measurement, "__tablename__")
    assert Measurement.__tablename__ == "measurements"


def test_measurement_attributes():
    """Test Measurement class attributes"""
    from src.models.db import Measurement

    assert hasattr(Measurement, "id")
    assert hasattr(Measurement, "task_id")
    assert hasattr(Measurement, "websdr_id")
    assert hasattr(Measurement, "frequency_mhz")
    assert hasattr(Measurement, "snr_db")
