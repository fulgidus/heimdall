# Prompt 05: Phase 5 - Frontend Training UI Components

## Context
GPU configured (Prompt 01 ✓), pipeline tested (Prompt 02 ✓), API built (Prompt 03 ✓), export format ready (Prompt 04 ✓). Final step: Build React frontend for training management.

## Current State
- Frontend: React + TypeScript + Vite at `frontend/`
- Existing: Dashboard, WebSDRs, Localization, Analytics, Sessions, Recordings, Settings, About
- Stack: Zustand state, shadcn/ui, Tailwind CSS, axios, WebSocket pattern working

## Architectural Decisions

### Page Structure
**Decision**: Training page with 3 tabs (Jobs, Metrics, Models)
**Location**: `frontend/src/pages/Training/`
**Why**: Separate job lifecycle, analysis, and artifacts

### State Management
**File**: `frontend/src/stores/trainingStore.ts` (NEW)
**Pattern**: Zustand store with jobs, metrics (Map<jobId, TrainingMetric[]>), models, wsConnected state
**Actions**: fetchJobs, fetchMetrics, fetchModels, createJob, cancelJob, pauseJob, resumeJob, downloadModel, importModel, handleJobUpdate, handleMetricUpdate

### Component Structure
```
frontend/src/pages/Training/
├── index.tsx                           # Main page with tabs
├── components/
│   ├── JobsTab/                        # Job list, cards, create dialog, details
│   ├── MetricsTab/                     # Charts (loss, accuracy, LR)
│   └── ModelsTab/                      # Model list, cards, export/import dialogs
├── hooks/                              # useTrainingJobs, useTrainingMetrics, useTrainingWebSocket, useModels
└── types.ts                            # TypeScript interfaces
```

### Real-Time Updates
**Pattern**: WebSocket at `ws://localhost:8001/ws/training/{jobId}` (same as WebSDR health)
**Events**: training_started, training_progress, training_completed, training_failed
**Handler**: Update store on message (handleJobUpdate/handleMetricUpdate)

### Charts
**Library**: Recharts (already in package.json)
**Components**: LossChart (train/val), AccuracyChart (train/val), LearningRateChart

## Tasks

### Task 1: TypeScript Types
**File**: `frontend/src/pages/Training/types.ts` (NEW)

**Interfaces**:
- `TrainingJob`: id, name, status (pending|running|paused|completed|failed|cancelled), created_at, started_at, completed_at, config, progress_percent, current_epoch, total_epochs, estimated_completion, error_message
- `TrainingJobConfig`: job_name, epochs, batch_size, learning_rate, model_architecture, dataset_id, validation_split, early_stopping_patience, checkpoint_every_n_epochs
- `TrainingMetric`: job_id, epoch, timestamp, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate, epoch_duration_seconds
- `TrainedModel`: id, name, version, architecture, created_at, training_job_id, onnx_path, parameters_count, final_metrics{train_loss, val_loss, train_accuracy, val_accuracy}
- `ExportOptions`: include_config, include_metrics, include_normalization, include_samples, num_samples, description

### Task 2: Training Store
**File**: `frontend/src/stores/trainingStore.ts` (NEW)

**Implementation**:
- State: jobs[], metrics Map, models[], wsConnected
- fetchJobs: GET /api/v1/training/jobs
- fetchMetrics: GET /api/v1/training/jobs/{jobId}/metrics, store in Map
- fetchModels: GET /api/v1/training/models
- createJob: POST /api/v1/training/jobs, return job_id
- cancelJob: DELETE /api/v1/training/jobs/{jobId}
- pauseJob: POST /api/v1/training/pause/{jobId}
- resumeJob: POST /api/v1/training/resume/{jobId}
- downloadModel: GET /api/v1/training/models/{modelId}/export with URLSearchParams, trigger blob download
- importModel: POST /api/v1/training/models/import with FormData
- handleJobUpdate: Update job in array by id
- handleMetricUpdate: Append metric to Map[job_id]

### Task 3: WebSocket Hook
**File**: `frontend/src/pages/Training/hooks/useTrainingWebSocket.ts` (NEW)

**Pattern**: 
- Connect to `ws://localhost:8001/ws/training/{jobId}` when jobId not null
- On message: Parse JSON, switch on event type (training_started/progress/completed/failed)
- Call handleJobUpdate or handleMetricUpdate from store
- On open/close: Update wsConnected state
- Cleanup: socket.close() on unmount

### Task 4: Jobs Tab
**Files**: 
- `frontend/src/pages/Training/components/JobsTab/JobsTab.tsx` (NEW)
- `frontend/src/pages/Training/components/JobsTab/JobCard.tsx` (NEW)
- `frontend/src/pages/Training/components/JobsTab/CreateJobDialog.tsx` (NEW)

**JobsTab**:
- Header with "New Training Job" button
- Grid of JobCards (3 columns on lg)
- Empty state message if no jobs
- Fetch jobs on mount + every 5s interval

**JobCard**:
- Card with job name, status badge (color-coded)
- Progress bar (current_epoch / total_epochs)
- Config details (architecture, batch_size, learning_rate)
- Action buttons: Pause (if running), Resume (if paused), Cancel (if running/paused)
- Use useTrainingWebSocket(job.id) if status === 'running'

**CreateJobDialog**:
- Form with fields: job_name, epochs, batch_size, learning_rate, model_architecture (dropdown)
- Optional: dataset_id (dropdown), validation_split, early_stopping_patience
- Submit calls createJob(), then close dialog

### Task 5: Metrics Tab
**Files**:
- `frontend/src/pages/Training/components/MetricsTab/MetricsTab.tsx` (NEW)
- `frontend/src/pages/Training/components/MetricsTab/LossChart.tsx` (NEW)
- `frontend/src/pages/Training/components/MetricsTab/AccuracyChart.tsx` (NEW)
- `frontend/src/pages/Training/components/MetricsTab/LearningRateChart.tsx` (NEW)

**MetricsTab**:
- Job selector dropdown (Select from shadcn/ui)
- When job selected: Fetch metrics on mount + every 5s
- Empty state if no job selected or no metrics yet
- Display 3 charts in vertical stack

**LossChart**: Recharts LineChart with train_loss (blue) and val_loss (red), XAxis=epoch, YAxis=loss

**AccuracyChart**: Recharts LineChart with train_accuracy (green) and val_accuracy (orange), XAxis=epoch, YAxis=accuracy

**LearningRateChart**: Recharts LineChart with learning_rate, XAxis=epoch, YAxis=LR (log scale)

### Task 6: Models Tab
**Files**:
- `frontend/src/pages/Training/components/ModelsTab/ModelsTab.tsx` (NEW)
- `frontend/src/pages/Training/components/ModelsTab/ModelCard.tsx` (NEW)
- `frontend/src/pages/Training/components/ModelsTab/ExportDialog.tsx` (NEW)
- `frontend/src/pages/Training/components/ModelsTab/ImportDialog.tsx` (NEW)

**ModelsTab**:
- Header with "Import Model" button
- Grid of ModelCards (3 columns on lg)
- Empty state if no models
- Fetch models on mount

**ModelCard**:
- Model name, version, architecture
- Created date, parameters count
- Final metrics (if available): train_loss, val_loss, train_accuracy, val_accuracy
- "Export" button opens ExportDialog

**ExportDialog**:
- Checkboxes: include_config, include_metrics, include_normalization, include_samples
- Number input: num_samples (1-10, default 5)
- Text input: description (optional)
- "Download" button calls downloadModel()

**ImportDialog**:
- File input (accept=".heimdall")
- Upload button calls importModel(file)
- Show progress/success/error toast

### Task 7: Main Training Page
**File**: `frontend/src/pages/Training/index.tsx` (NEW)

**Structure**:
- Tabs component from shadcn/ui with 3 tabs (Jobs, Metrics, Models)
- Tab content: JobsTab, MetricsTab, ModelsTab components

### Task 8: Add to Navigation
**Files**: 
- `frontend/src/App.tsx` (add route: `/training` → TrainingPage)
- `frontend/src/components/Navigation.tsx` (add link with Brain icon)

## Validation Criteria

### Must Pass
- [ ] Training page accessible from navigation
- [ ] Jobs tab displays all jobs from API
- [ ] Create job dialog works, triggers Celery task
- [ ] Job cards show real-time progress (WebSocket)
- [ ] Pause/resume/cancel buttons functional
- [ ] Metrics tab shows charts for selected job
- [ ] Charts update in real-time during training
- [ ] Models tab lists all trained models
- [ ] Export dialog downloads .heimdall file
- [ ] Import dialog accepts .heimdall file
- [ ] All components responsive (mobile + desktop)
- [ ] Error handling shows user-friendly messages

### UI/UX Requirements
- [ ] Progress bars animate smoothly
- [ ] Status badges use consistent colors (yellow=pending, blue=running, orange=paused, green=completed, red=failed, gray=cancelled)
- [ ] Loading states shown during API calls
- [ ] Confirmation dialogs for destructive actions (cancel/delete)
- [ ] Toast notifications for success/error
- [ ] Empty states have helpful messages

## Non-Breaking Requirement
- Existing pages (Dashboard, WebSDRs, etc.) unaffected
- No changes to existing routes
- Existing state management (other stores) untouched

## Success Criteria
When all validation passes, Phase 5 Training Pipeline is 100% COMPLETE. All prompts (01-05) executed successfully.

## Deliverables
Document:
- Screenshots of all 3 tabs (Jobs, Metrics, Models)
- WebSocket message log (real-time updates)
- Chart examples (loss, accuracy, LR)
- Export/import workflow test results
- Mobile responsiveness verification
- Any UI/UX improvements suggested

## Next Phase
Phase 5 complete → Proceed to **Phase 8: Kubernetes Deployment** (Phase 6 Inference already complete, Phase 7 Frontend in progress)
