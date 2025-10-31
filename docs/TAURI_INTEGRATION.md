# Tauri Desktop Integration Guide

## Overview

Heimdall includes a **Tauri desktop application layer** that wraps the existing React frontend without modifying any core code. The desktop app provides native features like GPU detection, local settings storage, and direct backend process management.

## Architecture

```
┌─────────────────────────────────────────────┐
│           React Frontend (Web)              │
│  - No changes to existing code              │
│  - Uses new tauri-bridge.ts utility         │
└────────────┬────────────────────────────────┘
             │ IPC Commands
             ▼
┌─────────────────────────────────────────────┐
│        Tauri Rust Backend                   │
│  - GPU detection (nvidia-smi)               │
│  - Settings storage (platform-specific)     │
│  - Data collection control                  │
│  - Training control                         │
└─────────────────────────────────────────────┘
```

## Quick Start

### Development Mode

```bash
# From project root
npm run tauri:dev
```

This will:
1. Start Vite dev server on http://localhost:3001
2. Open native Tauri window pointing to dev server
3. Enable hot reload for React changes

### Production Build

```bash
# Build frontend + create desktop executable
npm run build:app
```

Output locations:
- **Windows**: `tauri/target/release/bundle/msi/`
- **macOS**: `tauri/target/release/bundle/dmg/`
- **Linux**: `tauri/target/release/bundle/appimage/`

## Using Tauri Features in React

### 1. Import the Bridge

```typescript
import { tauriAPI, isTauriEnvironment } from '@/lib/tauri-bridge';
```

### 2. Check Environment

```typescript
function MyComponent() {
  const isDesktop = isTauriEnvironment();
  
  return (
    <div>
      Running in: {isDesktop ? 'Desktop' : 'Web'} mode
    </div>
  );
}
```

### 3. GPU Detection

```typescript
import { tauriAPI } from '@/lib/tauri-bridge';

async function checkGPU() {
  try {
    const gpuInfo = await tauriAPI.gpu.check();
    
    if (gpuInfo.available) {
      console.log(`GPU: ${gpuInfo.name}`);
      console.log(`Driver: ${gpuInfo.driver_version}`);
      console.log(`Memory: ${gpuInfo.memory_free_mb}MB free`);
      console.log(`Usage: ${gpuInfo.utilization_percent}%`);
    } else {
      console.log('No GPU detected');
    }
  } catch (error) {
    console.error('GPU check failed:', error);
  }
}
```

### 4. Settings Management

```typescript
import { tauriAPI, type AppSettings } from '@/lib/tauri-bridge';

// Load settings
async function loadSettings() {
  try {
    const settings = await tauriAPI.settings.load();
    console.log('API URL:', settings.api_url);
    console.log('GPU enabled:', settings.enable_gpu);
    return settings;
  } catch (error) {
    console.error('Failed to load settings:', error);
  }
}

// Save settings
async function saveSettings() {
  const newSettings: AppSettings = {
    api_url: 'http://localhost:8000',
    websocket_url: 'ws://localhost:80/ws',
    mapbox_token: 'your_token_here',
    auto_start_backend: false,
    backend_port: 8000,
    enable_gpu: true,
    theme: 'dark',
  };
  
  try {
    await tauriAPI.settings.save(newSettings);
    console.log('Settings saved successfully');
  } catch (error) {
    console.error('Failed to save settings:', error);
  }
}
```

### 5. Data Collection Control

```typescript
import { tauriAPI } from '@/lib/tauri-bridge';

async function startCollection() {
  try {
    const result = await tauriAPI.dataCollection.start({
      frequency: 145.500,
      duration_seconds: 60,
      websdrs: ['I8MOT', 'I2KBD', 'IW1QLH'],
    });
    console.log(result);
  } catch (error) {
    console.error('Collection start failed:', error);
  }
}

async function checkStatus() {
  try {
    const status = await tauriAPI.dataCollection.getStatus();
    console.log('Running:', status.is_running);
    console.log('Progress:', status.progress);
    console.log('Current WebSDR:', status.current_websdr);
  } catch (error) {
    console.error('Status check failed:', error);
  }
}
```

### 6. Training Control

```typescript
import { tauriAPI } from '@/lib/tauri-bridge';

async function startTraining() {
  try {
    const result = await tauriAPI.training.start({
      epochs: 100,
      batch_size: 32,
      learning_rate: 0.001,
      model_name: 'localization_model_v1',
    });
    console.log(result);
  } catch (error) {
    console.error('Training start failed:', error);
  }
}

async function monitorTraining() {
  try {
    const status = await tauriAPI.training.getStatus();
    console.log(`Epoch ${status.current_epoch}/${status.total_epochs}`);
    console.log(`Loss: ${status.loss.toFixed(4)}`);
    console.log(`Accuracy: ${(status.accuracy * 100).toFixed(2)}%`);
  } catch (error) {
    console.error('Training status check failed:', error);
  }
}
```

## Complete Example Component

```typescript
import React, { useState, useEffect } from 'react';
import { tauriAPI, isTauriEnvironment, type GpuInfo } from '@/lib/tauri-bridge';

export function DesktopFeatures() {
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [isDesktop, setIsDesktop] = useState(false);

  useEffect(() => {
    setIsDesktop(isTauriEnvironment());
    
    if (isTauriEnvironment()) {
      loadGPUInfo();
    }
  }, []);

  const loadGPUInfo = async () => {
    try {
      const info = await tauriAPI.gpu.check();
      setGpuInfo(info);
    } catch (error) {
      console.error('Failed to load GPU info:', error);
    }
  };

  if (!isDesktop) {
    return (
      <div>
        <p>Desktop-only features are not available in web mode.</p>
      </div>
    );
  }

  return (
    <div>
      <h2>Desktop Features</h2>
      
      {gpuInfo && (
        <div>
          <h3>GPU Information</h3>
          {gpuInfo.available ? (
            <>
              <p><strong>Name:</strong> {gpuInfo.name}</p>
              <p><strong>Driver:</strong> {gpuInfo.driver_version}</p>
              <p><strong>Memory:</strong> {gpuInfo.memory_free_mb}MB free / {gpuInfo.memory_total_mb}MB total</p>
              <p><strong>Usage:</strong> {gpuInfo.utilization_percent}%</p>
            </>
          ) : (
            <p>No GPU detected</p>
          )}
          <button onClick={loadGPUInfo}>Refresh GPU Info</button>
        </div>
      )}
    </div>
  );
}
```

## API Reference

### Environment Detection

- **`isTauriEnvironment(): boolean`** - Check if running in Tauri desktop mode

### Data Collection

- **`tauriAPI.dataCollection.start(config)`** - Start RF data collection
- **`tauriAPI.dataCollection.stop()`** - Stop ongoing collection
- **`tauriAPI.dataCollection.getStatus()`** - Get current status

### Training

- **`tauriAPI.training.start(config)`** - Start ML model training
- **`tauriAPI.training.stop()`** - Stop ongoing training
- **`tauriAPI.training.getStatus()`** - Get current training status

### GPU

- **`tauriAPI.gpu.check()`** - Check for GPU and get info
- **`tauriAPI.gpu.getUsage()`** - Get current GPU usage

### Settings

- **`tauriAPI.settings.load()`** - Load application settings
- **`tauriAPI.settings.save(settings)`** - Save application settings
- **`tauriAPI.settings.reset()`** - Reset to default settings

## Graceful Degradation

All Tauri commands will throw an error when called in web mode:

```typescript
try {
  await tauriAPI.gpu.check();
} catch (error) {
  // Error: "Tauri environment not available. This feature only works in desktop mode."
}
```

To handle this gracefully:

```typescript
if (isTauriEnvironment()) {
  // Use Tauri features
  const gpuInfo = await tauriAPI.gpu.check();
} else {
  // Web mode - use alternative or skip
  console.log('GPU detection not available in web mode');
}
```

## Platform-Specific Notes

### Windows
- GPU detection requires NVIDIA drivers and nvidia-smi
- Settings stored in: `%APPDATA%\heimdall-sdr\settings.json`

### macOS
- Settings stored in: `~/Library/Application Support/heimdall-sdr/settings.json`
- May require security approval for first run

### Linux
- GPU detection requires NVIDIA drivers and nvidia-smi in PATH
- Settings stored in: `~/.config/heimdall-sdr/settings.json`
- AppImage format for distribution

## Building for Production

### Prerequisites

**All Platforms:**
- Node.js 18+
- Rust 1.77+
- npm or pnpm

**Linux:**
```bash
sudo apt-get install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
```

**Windows:**
- Microsoft Visual C++ 2015-2022 Redistributable
- WebView2 (usually pre-installed on Windows 10/11)

**macOS:**
- Xcode Command Line Tools

### Build Commands

```bash
# Install all dependencies
npm run install:all

# Development mode
npm run tauri:dev

# Production build
npm run build:app

# Build frontend only
npm run build:frontend
```

## Troubleshooting

### Desktop app won't start
- Check that frontend is built: `cd frontend && npm run build`
- Verify Rust toolchain: `rustc --version`
- Check Tauri CLI: `npx tauri info`

### GPU not detected
- Verify nvidia-smi works: `nvidia-smi`
- Check driver installation
- Only NVIDIA GPUs supported currently

### Settings not persisting
- Check config directory exists and is writable
- Linux: `~/.config/heimdall-sdr/`
- Windows: `%APPDATA%\heimdall-sdr\`
- macOS: `~/Library/Application Support/heimdall-sdr/`

## Roadmap

Future desktop features:
- [ ] Auto-updater integration
- [ ] System tray integration
- [ ] Native notifications
- [ ] Background task execution
- [ ] AMD GPU support
- [ ] Local database option

## Contributing

When adding new Tauri commands:

1. Add Rust command in `tauri/src/commands/`
2. Register in `tauri/src/lib.rs`
3. Add TypeScript interface in `frontend/src/lib/tauri-bridge.ts`
4. Write unit tests in `frontend/src/lib/tauri-bridge.test.ts`
5. Update this documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines.
