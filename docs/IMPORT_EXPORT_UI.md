# Import/Export UI Overview

## Page Layout

The Import/Export page is located at **Settings → Import/Export** in the main menu.

### Page Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Import/Export Data                                          │
│ Save and restore your Heimdall SDR configuration and data  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ℹ️ Available Data                                           │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Sources: 10    WebSDRs: 7    Sessions: 5   Size: 11 KB │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌────────────────────────┬────────────────────────────────┐ │
│ │ 📥 Export Data         │ 📤 Import Data                 │ │
│ ├────────────────────────┼────────────────────────────────┤ │
│ │                        │                                │ │
│ │ Username *             │ [Load .heimdall File]          │ │
│ │ [________________]     │                                │ │
│ │                        │ ✅ File Loaded                 │ │
│ │ Full Name             │ Version: 1.0                   │ │
│ │ [________________]     │ Created: 2025-10-31 15:30     │ │
│ │                        │ Creator: operator1             │ │
│ │ Description           │                                │ │
│ │ [________________]     │ Import Sections                │ │
│ │ [________________]     │ ☑ Sources (10)                │ │
│ │                        │ ☑ WebSDRs (7)                 │ │
│ │ Include Sections       │ ☐ Sessions (5)                │ │
│ │ ☐ Settings            │                                │ │
│ │ ☑ Known Sources (10)  │ ☐ Overwrite existing data     │ │
│ │ ☑ WebSDRs (7)         │ Skip existing items if unchecked│ │
│ │ ☐ Recording Sessions  │                                │ │
│ │                        │ [Confirm Import]               │ │
│ │ [Export & Download]    │                                │ │
│ └────────────────────────┴────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## UI Components

### 1. Metadata Overview Section
- **Location**: Top of page, full width
- **Style**: Blue info alert box
- **Content**: 
  - Available data counts (sources, WebSDRs, sessions)
  - Estimated total file size
- **Purpose**: Shows user what data is available to export

### 2. Export Section (Left Column)

#### Creator Information
- **Username** (required): Text input
- **Full Name** (optional): Text input
- **Description** (optional): Textarea (2 rows)

#### Section Selection
- **Checkboxes** for each available section:
  - Settings
  - Known Sources (with count)
  - WebSDRs (with count)
  - Recording Sessions (with count)
  - Training Model (if available)
  - Inference Model (if available)

#### Export Button
- **Label**: "Export & Download"
- **Icon**: Download icon
- **States**:
  - Normal: Blue primary button
  - Loading: Shows spinner + "Exporting..."
  - Disabled: When username is missing

#### Status Messages
- **Success**: Green alert with checkmark
  - "Export successful! File downloaded."
- **Error**: Red alert with error icon
  - Shows specific error message

### 3. Import Section (Right Column)

#### File Selection
- **Button**: "Load .heimdall File"
- **Icon**: Upload icon
- **Action**: Opens browser file picker

#### File Preview (after loading)
- **Box Style**: Green success alert
- **Shows**:
  - File format version
  - Creation timestamp (formatted)
  - Creator username
  - Optional description

#### Section Selection
- **Dynamic checkboxes** based on file contents:
  - Only shows sections present in file
  - Each checkbox shows item count
  - Example: "Sources (10)"

#### Import Options
- **Overwrite existing**: Checkbox
  - Unchecked: Skip existing items (default)
  - Checked: Update existing items

#### Import Button
- **Label**: "Confirm Import"
- **Icon**: Upload icon
- **States**:
  - Normal: Green success button
  - Loading: Shows spinner + "Importing..."
  - Disabled: When no file loaded

#### Status Messages
- **Success**: Green alert with checkmark
  - Shows import counts per section
  - Lists any errors encountered
- **Error**: Red alert with error icon
  - Shows specific error message

## User Flows

### Export Flow
1. Navigate to Settings → Import/Export
2. Fill in username (required)
3. Optionally add name and description
4. Select sections to export using checkboxes
5. Click "Export & Download"
6. Browser downloads `heimdall-export-YYYY-MM-DD.heimdall`
7. Success message appears briefly

### Import Flow
1. Navigate to Settings → Import/Export
2. Click "Load .heimdall File"
3. Select file from computer
4. Review file metadata (version, creator, date)
5. Select which sections to import
6. Choose overwrite behavior
7. Click "Confirm Import"
8. Review import results (counts and errors)
9. Metadata overview updates automatically

## Visual Design

### Colors
- **Primary**: Blue buttons for exports
- **Success**: Green alerts and import button
- **Info**: Blue alert for metadata
- **Danger**: Red alerts for errors
- **Warning**: (not currently used)

### Icons
- **Download**: Export actions
- **Upload**: Import actions
- **Info**: Information alerts
- **AlertCircle**: Error messages
- **CheckCircle**: Success messages

### Layout
- **Two-column**: Export on left, Import on right
- **Responsive**: Works on mobile (stacks vertically)
- **Bootstrap 5**: Consistent with rest of application
- **Card-based**: Main content in card with header

### Spacing
- **Padding**: 4 units on container
- **Margins**: 3 units between form groups
- **Gap**: Standard Bootstrap spacing

## Accessibility

- **Labels**: All form fields have proper labels
- **Required fields**: Marked with asterisk (*)
- **Help text**: Provided for complex options
- **Loading states**: Clear visual feedback
- **Error messages**: Descriptive and actionable
- **Keyboard navigation**: All interactive elements accessible

## State Management

### Export States
- Initial: Empty form
- Loading: Button disabled, spinner shown
- Success: Green alert, auto-dismiss after 3s
- Error: Red alert, stays until dismissed

### Import States
- Initial: No file loaded, import button disabled
- File loaded: Preview shown, import button enabled
- Loading: Button disabled, spinner shown
- Success: Results shown with counts
- Error: Red alert with error details

## Data Validation

### Export Validation
- Username required before export
- At least one section must be selected
- API validates creator info format

### Import Validation
- File must be valid JSON
- File must match .heimdall schema
- Sections validated before import
- Foreign key relationships checked

## Performance

### Export
- Typical time: <1 second for 100 items
- File downloads immediately after generation
- No progress bar needed

### Import
- Typical time: <2 seconds for 100 items
- Shows spinner during processing
- Results appear after completion

## Error Handling

### Common Errors
- "Username is required"
- "Please load a .heimdall file first"
- "Invalid .heimdall file format"
- "Error importing source {name}: {details}"

### Error Display
- Red alert box with error icon
- Specific, actionable error messages
- Partial success shows both successes and errors
- Errors remain visible until cleared

## Integration

### Navigation
- Accessed via: Settings → Import/Export
- Menu icon: Download icon
- Protected route (requires authentication)

### API Integration
- Real-time metadata from `/api/import-export/export/metadata`
- Export via `/api/import-export/export`
- Import via `/api/import-export/import`
- Automatic retry on failure (via axios interceptor)

### State Updates
- Metadata refreshes after successful import
- No page reload needed
- WebSocket not used (RESTful operations)

## Browser Compatibility

### File Operations
- Uses HTML5 File API
- Creates download via blob URL
- File input for upload
- Works in all modern browsers

### Tested Browsers
- Chrome/Edge (Chromium)
- Firefox
- Safari
- Opera

## Mobile Responsiveness

### Layout Changes
- Two columns stack vertically on small screens
- Form inputs become full width
- Buttons remain full width
- Card padding reduces on mobile

### Touch Optimization
- Buttons sized for touch (minimum 44px)
- Adequate spacing between interactive elements
- File picker native to OS

## Future Enhancements

### Planned UI Improvements
1. Drag-and-drop file upload
2. Progress bar for large imports
3. Export history list
4. Quick export presets
5. File size preview before export
6. Validation report before import
7. Undo last import
8. Scheduled exports configuration

## Code Organization

### Files
- **Component**: `frontend/src/pages/ImportExport.tsx` (650 lines)
- **API Service**: `frontend/src/services/api/import-export.ts` (220 lines)
- **Route**: Registered in `frontend/src/App.tsx`
- **Menu**: Item in `frontend/src/components/layout/DattaLayout.tsx`

### Dependencies
- React hooks: useState, useEffect
- Lucide icons: Download, Upload, Info, AlertCircle, CheckCircle
- Bootstrap 5: Layout and styling
- Axios: API requests

### Patterns
- Functional component with hooks
- Form state management with useState
- API calls with async/await
- Error boundaries (inherited from layout)
- Loading states for all operations
