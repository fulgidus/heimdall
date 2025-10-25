# Map Component Architecture

## Component Hierarchy

```
Localization.tsx (Page)
    │
    ├── Statistics Cards (4 cards)
    │   ├── Active Receivers
    │   ├── Avg Accuracy
    │   ├── Confidence
    │   └── Signal Quality
    │
    ├── MapContainer (Main Map Component)
    │   │
    │   ├── useMapbox Hook
    │   │   ├── Initialize Mapbox GL JS
    │   │   ├── Add Navigation Controls
    │   │   ├── Add Fullscreen Control
    │   │   └── Add Scale Control
    │   │
    │   ├── WebSDRMarkers Component
    │   │   ├── Create DOM markers (7 receivers)
    │   │   ├── Apply status colors (green/yellow/red)
    │   │   ├── Add pulse animation (online receivers)
    │   │   └── Attach popups with metadata
    │   │
    │   └── LocalizationLayer Component
    │       ├── Create GeoJSON sources
    │       │   ├── Ellipses source (3 layers: 1σ, 2σ, 3σ)
    │       │   └── Points source (red circles)
    │       ├── Initialize Mapbox layers
    │       │   ├── Fill layer (3σ ellipse)
    │       │   ├── Fill layer (2σ ellipse)
    │       │   ├── Fill layer (1σ ellipse)
    │       │   └── Circle layer (points)
    │       ├── Update on data changes
    │       │   ├── Generate ellipse GeoJSON
    │       │   ├── Generate point GeoJSON
    │       │   └── Update map sources
    │       └── Handle interactions
    │           ├── Click handler → Show popup
    │           ├── Hover handler → Cursor pointer
    │           └── Auto-center on new points
    │
    └── Results Panel (Sidebar)
        ├── Recent Localizations List
        └── Selected Result Details
```

## Data Flow

```
WebSDR Store                Localization Store
     │                             │
     ├── websdrs[]                 ├── recentLocalizations[]
     └── healthStatus{}            └── currentPrediction
              │                             │
              │                             │
              └─────────────┬───────────────┘
                           │
                    Localization.tsx
                           │
                    MapContainer.tsx
                           │
              ┌────────────┴────────────┐
              │                         │
       WebSDRMarkers            LocalizationLayer
              │                         │
         DOM Markers               GeoJSON Layers
              │                         │
         Mapbox Map ←──────────────────┘
              │
         User Interactions
              │
         ┌────┴────┐
         │         │
      Popups    Clicks
         │         │
    Metadata   Callbacks
```

## Ellipse Calculation Flow

```
LocalizationResult
    │
    ├── latitude
    ├── longitude
    └── uncertainty_m
         │
         ▼
createCircularUncertainty()
         │
         ├── centerLat
         ├── centerLng
         ├── sigmaX = uncertainty_m
         ├── sigmaY = uncertainty_m
         └── rotation = 0
              │
              ▼
    createConfidenceEllipses()
         │
         ├── Level 1 (1σ - 68%)
         ├── Level 2 (2σ - 95%)
         └── Level 3 (3σ - 99.7%)
              │
              ▼
    generateEllipsePoints()
         │
         ├── For each confidence level:
         │   ├── Scale sigma by level
         │   ├── Generate 64 points
         │   │   └── Parametric ellipse equation
         │   ├── Apply rotation (0° for circular)
         │   └── Convert meters → degrees
         │        ├── Latitude: meters / 111320
         │        └── Longitude: meters / (111320 * cos(lat))
         │
         ▼
    GeoJSON Polygon Feature
         │
         ├── type: 'Feature'
         ├── geometry:
         │   ├── type: 'Polygon'
         │   └── coordinates: [[lng, lat], ...]
         └── properties:
             ├── confidenceLevel: 1|2|3
             ├── centerLat
             ├── centerLng
             └── ...
              │
              ▼
    Mapbox Layer (Fill)
         │
         ├── paint:
         │   ├── fill-color: getConfidenceColor(level)
         │   └── fill-opacity: getConfidenceOpacity(level)
         └── filter: ['==', ['get', 'confidenceLevel'], level]
```

## State Management

```
Component State (useRef)
    │
    ├── mapRef: mapboxgl.Map | null
    ├── markersRef: mapboxgl.Marker[]
    ├── popupsRef: mapboxgl.Popup[]
    └── layersInitialized: boolean

Hook State (useState)
    │
    ├── isLoaded: boolean
    └── error: string | null

Store State (Zustand)
    │
    ├── websdrs: WebSDRConfig[]
    ├── healthStatus: Record<number, WebSDRHealthStatus>
    └── recentLocalizations: LocalizationResult[]

Props
    │
    ├── onLocalizationClick?: (loc) => void
    ├── style?: CSSProperties
    └── className?: string
```

## Lifecycle Events

```
Component Mount
    │
    ▼
useMapbox Hook
    │
    ├── Get Mapbox token
    ├── Initialize map
    ├── Add controls
    ├── Set event listeners
    │   ├── 'load' → setIsLoaded(true)
    │   └── 'error' → setError(message)
    │
    ▼
Map Loaded
    │
    ├── WebSDRMarkers useEffect
    │   ├── Clear existing markers
    │   ├── Create new markers
    │   └── Add to map
    │
    ├── LocalizationLayer useEffect
    │   ├── initializeLayers() (once)
    │   │   ├── Add GeoJSON sources
    │   │   ├── Add fill layers (ellipses)
    │   │   ├── Add circle layer (points)
    │   │   └── Add event handlers
    │   │
    │   └── updateLocalizationData() (on changes)
    │       ├── Generate ellipse features
    │       ├── Generate point features
    │       ├── Update GeoJSON sources
    │       └── Auto-center if needed
    │
    ▼
User Interaction
    │
    ├── Click WebSDR marker
    │   └── Show popup (metadata)
    │
    ├── Click localization point
    │   ├── Show popup (details)
    │   └── Call onLocalizationClick()
    │
    └── Hover localization point
        └── Change cursor to pointer

Component Unmount
    │
    ├── Remove all markers
    ├── Remove all popups
    └── Remove map instance
```

## Performance Optimization

```
Rendering Strategy
    │
    ├── WebSDR Markers (7 receivers)
    │   └── DOM markers (manageable count)
    │       └── Custom div elements with CSS
    │
    └── Localizations (up to 100 points)
        └── GeoJSON layers (GPU-accelerated)
            ├── Ellipses: Fill layers
            └── Points: Circle layer

Re-rendering Control
    │
    ├── useCallback (stable references)
    │   ├── initializeLayers
    │   └── updateLocalizationData
    │
    ├── useRef (prevent re-renders)
    │   ├── mapRef
    │   ├── markersRef
    │   ├── popupsRef
    │   └── layersInitialized
    │
    └── useEffect dependencies
        ├── [map, websdrs, healthStatus]
        └── [initializeLayers, updateLocalizationData]

Data Updates
    │
    ├── WebSDRs changed
    │   └── Recreate all markers
    │
    └── Localizations changed
        └── Update GeoJSON sources
            └── Mapbox handles rendering
```

## Error Handling

```
Error Sources
    │
    ├── Missing Mapbox token
    │   └── Show error message with instructions
    │
    ├── Map initialization failure
    │   └── Catch in useMapbox, set error state
    │
    ├── Map load error
    │   └── 'error' event → setError(message)
    │
    └── Runtime errors
        └── Console.error, graceful degradation
```

## File Organization

```
frontend/src/
    │
    ├── components/Map/
    │   ├── MapContainer.tsx       (Main wrapper)
    │   ├── WebSDRMarkers.tsx      (Receiver markers)
    │   ├── LocalizationLayer.tsx  (Localization display)
    │   ├── Map.css                (Animations, styles)
    │   ├── README.md              (Component docs)
    │   └── index.ts               (Barrel exports)
    │
    ├── hooks/
    │   └── useMapbox.ts           (Map hook)
    │
    ├── utils/
    │   ├── ellipse.ts             (Ellipse math)
    │   └── ellipse.test.ts        (16 tests)
    │
    └── pages/
        └── Localization.tsx       (Page integration)
```

This architecture ensures:
- ✅ Clean separation of concerns
- ✅ Efficient rendering (GeoJSON for performance)
- ✅ Proper React patterns (hooks, refs, memoization)
- ✅ Testable code (pure functions for ellipse calc)
- ✅ Maintainable structure (logical file organization)
