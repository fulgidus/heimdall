# Import/Export Feature Documentation

## Overview

The Import/Export feature allows users to save and restore their Heimdall SDR system state using structured `.heimdall` JSON files. This enables:

- **Backup & Restore**: Regular backups of configuration and data
- **Data Migration**: Moving between installations or environments
- **Configuration Sharing**: Share WebSDR configurations with team members
- **Disaster Recovery**: Quick restoration from saved state

## File Format

### .heimdall File Structure

The `.heimdall` file is a JSON file with the following structure:

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2025-10-31T15:30:00Z",
    "creator": {
      "username": "operator1",
      "name": "John Doe"
    },
    "section_sizes": {
      "settings": 256,
      "sources": 2048,
      "websdrs": 1536,
      "sessions": 0,
      "training_model": 0,
      "inference_model": 0
    },
    "description": "Weekly backup - October 2025"
  },
  "sections": {
    "settings": { ... },
    "sources": [ ... ],
    "websdrs": [ ... ],
    "sessions": [ ... ],
    "training_model": { ... },
    "inference_model": { ... }
  }
}
```

### Metadata Section

- **version**: File format version (currently "1.0")
- **created_at**: ISO 8601 timestamp of export
- **creator**: Information about who created the export
  - **username**: Username (required)
  - **name**: Full name (optional)
- **section_sizes**: Byte sizes of each section
- **description**: Optional description of the export

### Data Sections

#### Settings Section
User preferences and application settings.

```json
{
  "theme": "light",
  "default_frequency_mhz": 145.5,
  "default_duration_seconds": 10.0,
  "auto_approve_sessions": false
}
```

#### Sources Section
Known RF sources for training and localization.

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Test Beacon",
    "description": "VHF beacon for testing",
    "frequency_hz": 145500000,
    "latitude": 45.5,
    "longitude": 9.2,
    "power_dbm": 10.0,
    "source_type": "beacon",
    "is_validated": true,
    "error_margin_meters": 30.0,
    "created_at": "2025-10-31T10:00:00Z",
    "updated_at": "2025-10-31T10:00:00Z"
  }
]
```

#### WebSDRs Section
WebSDR receiver configurations.

```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440000",
    "name": "F5LEN Toulouse",
    "url": "http://websdr.f5len.net:8901",
    "location_description": "Toulouse, France",
    "latitude": 43.5,
    "longitude": 1.4,
    "altitude_meters": 150.0,
    "country": "France",
    "operator": "F5LEN",
    "is_active": true,
    "timeout_seconds": 30,
    "retry_count": 3,
    "created_at": "2025-10-31T10:00:00Z",
    "updated_at": "2025-10-31T10:00:00Z"
  }
]
```

#### Sessions Section
Recording session metadata.

```json
[
  {
    "id": "770e8400-e29b-41d4-a716-446655440000",
    "known_source_id": "550e8400-e29b-41d4-a716-446655440000",
    "session_name": "Training Session 1",
    "session_start": "2025-10-31T10:00:00Z",
    "session_end": "2025-10-31T10:10:00Z",
    "duration_seconds": 600.0,
    "celery_task_id": "task-123",
    "status": "completed",
    "approval_status": "approved",
    "notes": "Good quality data",
    "created_at": "2025-10-31T10:00:00Z",
    "updated_at": "2025-10-31T10:15:00Z",
    "measurements_count": 42
  }
]
```

## API Endpoints

### GET /api/import-export/export/metadata

Get metadata about available data for export.

**Response:**
```json
{
  "sources_count": 10,
  "websdrs_count": 7,
  "sessions_count": 5,
  "has_training_model": false,
  "has_inference_model": false,
  "estimated_sizes": {
    "settings": 256,
    "sources": 5000,
    "websdrs": 2800,
    "sessions": 3000,
    "training_model": 0,
    "inference_model": 0
  }
}
```

### POST /api/import-export/export

Export selected data sections to .heimdall format.

**Request:**
```json
{
  "creator": {
    "username": "operator1",
    "name": "John Doe"
  },
  "description": "Weekly backup",
  "include_settings": false,
  "include_sources": true,
  "include_websdrs": true,
  "include_sessions": false,
  "include_training_model": false,
  "include_inference_model": false
}
```

**Response:**
```json
{
  "file": {
    "metadata": { ... },
    "sections": { ... }
  },
  "size_bytes": 8192
}
```

### POST /api/import-export/import

Import data from .heimdall file format.

**Request:**
```json
{
  "heimdall_file": {
    "metadata": { ... },
    "sections": { ... }
  },
  "import_settings": false,
  "import_sources": true,
  "import_websdrs": true,
  "import_sessions": false,
  "import_training_model": false,
  "import_inference_model": false,
  "overwrite_existing": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Import completed successfully",
  "imported_counts": {
    "settings": 0,
    "sources": 10,
    "websdrs": 7,
    "sessions": 0,
    "training_model": 0,
    "inference_model": 0
  },
  "errors": []
}
```

## Usage

### Web Application

#### Exporting Data

1. Navigate to **Settings → Import/Export** in the menu
2. Fill in the **Creator Info**:
   - Username (required)
   - Full Name (optional)
   - Description (optional)
3. Select sections to export using checkboxes:
   - Settings
   - Known Sources
   - WebSDRs
   - Recording Sessions
4. Click **Export & Download**
5. File will be downloaded as `heimdall-export-YYYY-MM-DD.heimdall`

#### Importing Data

1. Navigate to **Settings → Import/Export** in the menu
2. Click **Load .heimdall File**
3. Select a `.heimdall` file from your computer
4. Review the file metadata and contents
5. Select sections to import using checkboxes
6. Choose whether to **Overwrite existing data**:
   - Checked: Updates existing items with same ID
   - Unchecked: Skips existing items
7. Click **Confirm Import**
8. Review the import results

### Command Line (curl)

#### Export Example

```bash
curl -X POST http://localhost:8001/api/import-export/export \
  -H "Content-Type: application/json" \
  -d '{
    "creator": {
      "username": "admin",
      "name": "System Admin"
    },
    "description": "Automated backup",
    "include_sources": true,
    "include_websdrs": true
  }' \
  > backup-$(date +%Y%m%d).heimdall
```

#### Import Example

```bash
curl -X POST http://localhost:8001/api/import-export/import \
  -H "Content-Type: application/json" \
  -d @import-request.json
```

Where `import-request.json` contains:
```json
{
  "heimdall_file": {
    "metadata": { ... },
    "sections": { ... }
  },
  "import_sources": true,
  "import_websdrs": true,
  "overwrite_existing": false
}
```

## Use Cases

### Regular Backups

Export all data weekly for disaster recovery:
```json
{
  "include_settings": true,
  "include_sources": true,
  "include_websdrs": true,
  "include_sessions": true
}
```

### Configuration Sharing

Share WebSDR configurations with team members:
```json
{
  "include_settings": false,
  "include_sources": false,
  "include_websdrs": true,
  "include_sessions": false
}
```

### Data Migration

Move to a new installation:
```json
{
  "include_settings": true,
  "include_sources": true,
  "include_websdrs": true,
  "include_sessions": false
}
```

### Source Library Export

Share known RF sources:
```json
{
  "include_settings": false,
  "include_sources": true,
  "include_websdrs": false,
  "include_sessions": false
}
```

## File Size Estimates

Typical file sizes for different data volumes:

| Section | Items | Estimated Size |
|---------|-------|---------------|
| Settings | 1 | 256 bytes |
| Sources | 10 | ~5 KB |
| Sources | 100 | ~50 KB |
| WebSDRs | 7 | ~3 KB |
| WebSDRs | 20 | ~8 KB |
| Sessions | 10 | ~6 KB |
| Sessions | 100 | ~60 KB |

Complete export with 100 sources, 20 WebSDRs, 100 sessions: ~120 KB

## Best Practices

### Export

1. **Add descriptions**: Help identify exports later
2. **Regular schedule**: Export weekly or after major changes
3. **Version control**: Use dated filenames
4. **Selective exports**: Only include needed sections
5. **Validate creator info**: Use consistent usernames

### Import

1. **Preview first**: Review metadata before importing
2. **Test in staging**: Import to test environment first
3. **Backup current data**: Export before importing
4. **Use overwrite carefully**: Understand the impact
5. **Check import results**: Review imported counts and errors

### Security

1. **Protect files**: .heimdall files contain configuration data
2. **Review before sharing**: Don't share sensitive configurations
3. **Validate sources**: Only import from trusted sources
4. **Monitor imports**: Review import results for anomalies

## Troubleshooting

### Export Issues

**Problem**: "Username is required"
- **Solution**: Fill in the username field before exporting

**Problem**: Export returns no data
- **Solution**: Check that you have data in the selected sections

**Problem**: Export is too large
- **Solution**: Export sections individually or reduce data volume

### Import Issues

**Problem**: "Invalid .heimdall file format"
- **Solution**: Ensure file is valid JSON and matches schema

**Problem**: Import partially succeeds
- **Solution**: Review error messages, check for ID conflicts

**Problem**: Overwrite not working as expected
- **Solution**: Verify IDs match between import and existing data

**Problem**: Foreign key violations
- **Solution**: Import in correct order (sources before sessions)

## Technical Details

### Database Transactions

All import operations use database transactions to ensure atomicity:
- Complete success: All data imported
- Complete failure: No data imported, rollback performed
- Partial success: Valid items imported, errors reported

### Conflict Resolution

Import behavior with `overwrite_existing`:

| overwrite_existing | Existing ID | Behavior |
|-------------------|-------------|----------|
| `false` | Exists | Skip, continue |
| `false` | Not exists | Insert |
| `true` | Exists | Update |
| `true` | Not exists | Insert |

### Data Validation

Imports validate:
- File format and schema
- Required fields
- Data types
- Constraint violations
- Foreign key relationships

### Performance

- Export: O(n) where n is number of items
- Import: O(n) with batch operations
- Typical times:
  - Export 100 items: <1 second
  - Import 100 items: <2 seconds

## Future Enhancements

Planned features (not yet implemented):

1. **Compression**: GZIP compression for large exports
2. **Encryption**: AES encryption for sensitive data
3. **Incremental exports**: Export only changes since last export
4. **Model files**: Include ML model weights in exports
5. **Import history**: Track all import operations
6. **Scheduled exports**: Automatic periodic exports
7. **Cloud sync**: Sync to cloud storage
8. **Validation report**: Detailed pre-import validation

## Support

For issues or questions:
- Check troubleshooting section above
- Review API endpoint documentation
- Examine import error messages
- Check backend logs for details
- Contact system administrator

## Version History

- **v1.0** (2025-10-31): Initial release
  - Basic export/import functionality
  - Support for sources, WebSDRs, sessions
  - Web interface
  - API endpoints
