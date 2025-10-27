-- Quick seed for WebSDR stations using hard-coded Italian receivers.
-- This bypasses the health-check API and seeds with known-good data.
SET
    search_path TO heimdall,
    public;

-- Insert Italian WebSDR stations with known coordinates
INSERT INTO
    websdr_stations (
        name,
        url,
        country,
        latitude,
        longitude,
        is_active,
        timeout_seconds,
        retry_count,
        location_description
    )
VALUES
    (
        'Aquila di Giaveno',
        'http://sdr1.ik1jns.it:8076',
        'Italy',
        45.03,
        7.27,
        TRUE,
        30,
        3,
        'Aquila, Giaveno, Turin'
    ),
    (
        'Montanaro',
        'http://cbfenis.ddns.net:43510',
        'Italy',
        45.234,
        7.857,
        TRUE,
        30,
        3,
        'Montanaro, Italy'
    ),
    (
        'Torino',
        'http://vst-aero.it:8073',
        'Italy',
        45.044,
        7.672,
        TRUE,
        30,
        3,
        'Torino, Italy'
    ),
    (
        'Coazze',
        'http://94.247.189.130:8076',
        'Italy',
        45.03,
        7.27,
        TRUE,
        30,
        3,
        'Coazze, Italy'
    ),
    (
        'Passo del Giovi',
        'http://iz1mlt.ddns.net:8074',
        'Italy',
        44.561,
        8.956,
        TRUE,
        30,
        3,
        'Passo del Giovi, Italy'
    ),
    (
        'Genova',
        'http://iq1zw.ddns.net:42154',
        'Italy',
        44.395,
        8.956,
        TRUE,
        30,
        3,
        'Genova, Italy'
    ),
    (
        'Milano - Baggio',
        'http://iu2mch.duckdns.org:8073',
        'Italy',
        45.478,
        9.123,
        TRUE,
        30,
        3,
        'Milano (Baggio), Italy'
    ) ON CONFLICT (name) DO
UPDATE
SET
    url = EXCLUDED.url,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    is_active = EXCLUDED.is_active,
    timeout_seconds = EXCLUDED.timeout_seconds,
    retry_count = EXCLUDED.retry_count,
    location_description = EXCLUDED.location_description,
    updated_at = NOW();