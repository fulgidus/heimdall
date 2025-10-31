# Import/Export Feature Guide

## Overview

The Import/Export feature allows users to save and restore their Heimdall SDR progress across different installations and build modes. Data is stored in `.heimdall` files using a structured JSON format.

## Key Features

- **Selective Export**: Choose which data sections to include
- **Cross-Platform**: Works in Desktop (Training/Inference) and Web (Docker) modes
- **Data Integrity**: Complete validation on import
- **Human-Readable**: JSON format for easy inspection and editing

## .heimdall File Format

### Structure

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2025-10-31T15:00:00Z",
    "creator": {
      "username": "operator1",
      "name": "John Doe"
    },
    "section_sizes": {
      "settings": 1024,
      "sources": 2048,
      "websdrs": 1536,
      "sessions": 5120,
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

### Metadata (Required)

- **version**: File format version (currently "1.0")
- **created_at**: ISO 8601 timestamp
- **creator**: Username and full name
- **section_sizes**: Size of each section in bytes
- **description**: Optional description

### Sections (Optional)

Each section can be independently included or excluded:

1. **settings**: User preferences (theme, language, defaults)
2. **sources**: Known RF sources with coordinates
3. **websdrs**: WebSDR receiver configurations
4. **sessions**: Recording sessions with metadata
5. **training_model**: ML model in training
6. **inference_model**: Production ML model

## Using Import/Export

### Desktop Application (Tauri)

#### File Association

When you install the Heimdall SDR desktop application, it automatically registers as the default application for `.heimdall` files with your operating system.

**Opening .heimdall Files:**
- **Double-click** any `.heimdall` file in your file explorer
- The application will launch automatically
- You'll be taken directly to the **Import/Export** page
- The file will be loaded and ready for import

This makes it easy to share configurations with colleagues or restore backups - just double-click the file!

#### Exporting Data

1. Navigate to **Settings** → **Import/Export**
2. Fill in creator information (username, name)
3. Optionally add a description
4. Select sections to export:
   - ✅ User Settings
   - ✅ Known Sources
   - ✅ WebSDR Stations
   - ✅ Recording Sessions
   - ✅ Training Model (if available)
   - ✅ Inference Model (if available)
5. Click **Export Data**
6. Choose save location in file dialog
7. File is saved as `heimdall-export-YYYY-MM-DD.heimdall`

#### Importing Data

**Method 1: Via Application**
1. Navigate to **Settings** → **Import/Export**
2. Click **Open .heimdall File**
3. Select file from file dialog
4. Review file metadata and available sections
5. Select sections to import
6. Choose whether to overwrite existing data
7. Click **Confirm Import**
8. Review import results

**Method 2: Double-Click File (Recommended)**
1. Double-click the `.heimdall` file in your file explorer
2. Application opens to Import/Export page with file pre-loaded
3. Review file metadata and available sections
4. Select sections to import
5. Choose whether to overwrite existing data
6. Click **Confirm Import**
7. Review import results

### Web Application (Docker)

#### Exporting Data

1. Navigate to **Settings** → **Import/Export**
2. Fill in creator information
3. Select sections to export
4. Click **Export Data**
5. Browser downloads the `.heimdall` file

#### Importing Data

1. Navigate to **Settings** → **Import/Export**
2. Click **Choose .heimdall File**
3. Select file from your computer
4. Review and confirm import options
5. Click **Confirm Import**

## Build Modes

### Training Mode

Full-featured build with local training capabilities:

```bash
npm run build:app:training
```

Features:
- All import/export sections
- Training model support
- Full dataset management

### Inference Mode

Lightweight build for inference-only operations:

```bash
npm run build:app:inference
```

Features:
- Import/export of sources and WebSDRs
- Inference model only
- No training capabilities

### Docker Mode

Web-based version running in containers:

```bash
docker-compose up
```

Features:
- All import/export via browser
- Shared backend with desktop modes
- Network accessible

## API Endpoints

### Export Data

```http
POST /api/import-export/export
Content-Type: application/json

{
  "creator": {
    "username": "operator1",
    "name": "John Doe"
  },
  "description": "Weekly backup",
  "include_settings": true,
  "include_sources": true,
  "include_websdrs": true,
  "include_sessions": false,
  "include_training_model": false,
  "include_inference_model": false
}
```

Response: Complete `.heimdall` file as JSON

### Import Data

```http
POST /api/import-export/import
Content-Type: application/json

{
  "file_content": { ... },
  "import_settings": true,
  "import_sources": true,
  "import_websdrs": true,
  "import_sessions": false,
  "overwrite_existing": false
}
```

Response:

```json
{
  "success": true,
  "message": "Import completed successfully",
  "imported_counts": {
    "sources": 10,
    "websdrs": 7,
    "settings": 1
  },
  "errors": [],
  "warnings": []
}
```

### Get Export Metadata

```http
GET /api/import-export/export/metadata
```

Response:

```json
{
  "available_sources_count": 10,
  "available_websdrs_count": 7,
  "available_sessions_count": 5,
  "has_training_model": false,
  "has_inference_model": true,
  "estimated_size_bytes": 15360
}
```

## Tauri Commands

### Save File

```typescript
import { invoke } from '@tauri-apps/api/core';

const result = await invoke('save_heimdall_file', {
  content: JSON.stringify(heimdallFile),
  defaultFilename: 'backup.heimdall',
});
```

### Load File

```typescript
const result = await invoke('load_heimdall_file');
if (result.success) {
  const heimdallFile = JSON.parse(result.content);
}
```

### Load from Path

```typescript
const result = await invoke('load_heimdall_file_from_path', {
  path: '/path/to/file.heimdall',
});
```

### Save to Path

```typescript
const result = await invoke('save_heimdall_file_to_path', {
  content: JSON.stringify(heimdallFile),
  path: '/path/to/file.heimdall',
});
```

## File Association Implementation

### OS Registration

The application registers itself as the default handler for `.heimdall` files through Tauri's bundle configuration:

```json
{
  "bundle": {
    "fileAssociations": [
      {
        "ext": ["heimdall"],
        "name": "Heimdall Export File",
        "description": "Heimdall SDR configuration and data export file",
        "role": "Editor",
        "mimeType": "application/x-heimdall"
      }
    ]
  }
}
```

This configuration is included in all build modes (standard, training, inference).

### Command-Line Handling

When the OS launches the application with a `.heimdall` file, Tauri captures the file path as a command-line argument:

1. Application starts with file path as argument
2. Backend detects `.heimdall` file in arguments
3. Emits `open-heimdall-file` event to frontend
4. Frontend listens for event and automatically:
   - Navigates to Import/Export page
   - Loads the file content
   - Displays import preview

### Event Flow

```typescript
// Backend (Rust) - lib.rs
if arg.ends_with(".heimdall") {
  app_handle.emit("open-heimdall-file", file_path);
}

// Frontend (TypeScript) - App.tsx
listen('open-heimdall-file', (event) => {
  navigate('/import-export');
});

// Frontend (TypeScript) - ImportExport.tsx
listen('open-heimdall-file', async (event) => {
  const filePath = event.payload;
  const result = await invoke('load_heimdall_file_from_path', { path: filePath });
  // Load and display file for import
});
```

## Best Practices

### Regular Backups

- Export weekly after data collection sessions
- Include sessions only if they're approved
- Add descriptive metadata for easy identification

### Version Control

- Keep multiple export versions
- Use descriptive filenames: `heimdall-2025-10-31-weekly.heimdall`
- Document what changed between versions

### Data Migration

- Test import on non-production instance first
- Use overwrite carefully - backup current data first
- Review import warnings and errors

### File Management

- Store `.heimdall` files in secure location
- Compress for long-term storage (JSON compresses well)
- Version control important configurations

## Troubleshooting

### Import Fails

**Problem**: "Invalid .heimdall file format"

**Solution**: 
- Verify JSON syntax with validator
- Check file wasn't corrupted
- Ensure version compatibility

**Problem**: "Import completed with errors"

**Solution**:
- Check error messages in result
- Verify IDs don't conflict
- Use overwrite mode if needed

### Export Fails

**Problem**: "Export failed: database error"

**Solution**:
- Check backend connectivity
- Verify database is accessible
- Review backend logs

### Missing Sections

**Problem**: Sections not available for import

**Solution**:
- Verify sections were exported
- Check file metadata.section_sizes
- Re-export with correct options

## Security Considerations

- `.heimdall` files contain sensitive data
- Don't share files with untrusted parties
- Sanitize before sharing publicly
- Use encryption for sensitive deployments

## Examples

### Full Backup

```bash
# Export everything
curl -X POST http://localhost:8001/api/import-export/export \
  -H "Content-Type: application/json" \
  -d '{
    "creator": {"username": "admin", "name": "Admin User"},
    "description": "Full system backup",
    "include_settings": true,
    "include_sources": true,
    "include_websdrs": true,
    "include_sessions": true,
    "include_training_model": true,
    "include_inference_model": true
  }' > backup-full.heimdall
```

### Minimal Configuration Export

```bash
# Export only sources and WebSDRs
curl -X POST http://localhost:8001/api/import-export/export \
  -H "Content-Type: application/json" \
  -d '{
    "creator": {"username": "admin", "name": "Admin User"},
    "description": "Configuration only",
    "include_settings": false,
    "include_sources": true,
    "include_websdrs": true,
    "include_sessions": false
  }' > config-only.heimdall
```

## File Size Estimates

Typical section sizes:

- **Settings**: ~1 KB
- **Sources**: ~1 KB per source
- **WebSDRs**: ~2 KB per station
- **Sessions**: ~5 KB per session (without measurements)
- **Training Model**: Variable (can be large)
- **Inference Model**: Variable (optimized ONNX)

Example: 10 sources + 7 WebSDRs + 5 sessions = ~40 KB

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System architecture
- [API](API.md) - API reference
- [Development](DEVELOPMENT.md) - Development guide
