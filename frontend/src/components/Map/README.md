# Map Components

Interactive Mapbox GL JS map components for displaying WebSDR receivers and RF source localizations.

## Components

### MapContainer

Main map container component that integrates all map layers.

**Props:**
- `websdrs: WebSDRConfig[]` - Array of WebSDR receiver configurations
- `healthStatus: Record<number, WebSDRHealthStatus>` - Health status for each WebSDR
- `localizations: LocalizationResult[]` - Array of localization results to display
- `onLocalizationClick?: (localization: LocalizationResult) => void` - Click handler for localization points
- `style?: React.CSSProperties` - Custom styles for the map container
- `className?: string` - Additional CSS classes

**Example:**
```tsx
import { MapContainer } from '@/components/Map';

<MapContainer
  websdrs={websdrs}
  healthStatus={healthStatus}
  localizations={recentLocalizations}
  onLocalizationClick={(loc) => setSelectedLocalization(loc)}
  style={{ height: '500px' }}
/>
```

### WebSDRMarkers

Displays WebSDR receiver locations with status-based colors.

**Features:**
- Green markers for online receivers
- Red markers for offline receivers
- Yellow markers for unknown status
- Pulse animation for online receivers
- Hover tooltips with receiver info
- Click to show detailed popup

### LocalizationLayer

Displays localization results with uncertainty ellipses.

**Features:**
- Red circle markers for localization points
- 3-level uncertainty ellipses (1σ, 2σ, 3σ)
- Color-coded by confidence level:
  - Green: 68% confidence (1-sigma)
  - Yellow/Orange: 95% confidence (2-sigma)
  - Light Blue: 99.7% confidence (3-sigma)
- Auto-centering on new localizations
- Maximum 100 historical points (configurable via `maxPoints` prop)

## Utilities

### ellipse.ts

Utility functions for calculating uncertainty ellipses.

**Key Functions:**
- `generateEllipsePoints(params, numPoints)` - Generate ellipse coordinates
- `createEllipseFeature(params, properties)` - Create GeoJSON Feature
- `createConfidenceEllipses(params, levels)` - Create multiple confidence levels
- `getConfidenceColor(level)` - Get color for confidence level
- `createCircularUncertainty(lat, lng, radius)` - Create circular uncertainty

## Setup

### 1. Get Mapbox Access Token

1. Create account at https://www.mapbox.com/
2. Generate access token
3. Add to `.env` file:

```bash
VITE_MAPBOX_TOKEN=pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJjbHh4eHh4eHgifQ.xxxxxxxxxxxx
```

### 2. Import Components

```tsx
import { MapContainer } from '@/components/Map';
import type { WebSDRConfig, LocalizationResult } from '@/services/api/types';
```

### 3. Use in Page

```tsx
<MapContainer
  websdrs={websdrs}
  healthStatus={healthStatus}
  localizations={localizations}
  onLocalizationClick={(loc) => console.log('Clicked:', loc)}
  style={{ height: '600px' }}
/>
```

## Error Handling

If Mapbox token is not configured, the component displays a friendly error message:

```
⚠️ Map Error: Mapbox access token not configured.
Please configure VITE_MAPBOX_TOKEN in your .env file.
```

## Performance

- Uses GeoJSON sources for efficient rendering
- GPU-accelerated ellipse rendering
- React.memo and useCallback for optimal re-rendering
- Automatic cleanup on component unmount
- Efficient popup management

## Testing

Run tests:
```bash
npm test src/utils/ellipse.test.ts
```

16 tests covering:
- Ellipse point generation
- GeoJSON Feature creation
- Confidence level calculations
- Color and opacity mappings
- Circular uncertainty creation

## Future Enhancements

- Layer control panel for toggling visibility
- Full ellipse with rotation support (currently circular)
- Heat map for detection density
- Timestamp slider for historical playback
- Trail lines connecting chronological points
- Custom marker clustering for 500+ points
