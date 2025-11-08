/**
 * Sample Set Range Selector Component
 *
 * Allows users to select a specific range of samples from a dataset for export.
 * Features:
 * - Enable/disable full dataset export
 * - Slider for visual range selection
 * - Number inputs for precise offset/limit control
 * - Real-time size estimation
 */

import { useState, useEffect } from 'react';
import type { AvailableSampleSet } from '@/services/api/import-export';

interface SampleSetRangeSelectorProps {
  sampleSet: AvailableSampleSet;
  enabled: boolean;
  range: { offset: number; limit: number | null };
  onEnabledChange: (enabled: boolean) => void;
  onRangeChange: (offset: number, limit: number | null) => void;
}

/**
 * Format bytes to human-readable format
 */
function formatBytes(bytes: number): string {
  // Handle edge cases
  if (!bytes || !isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export default function SampleSetRangeSelector({
  sampleSet,
  enabled,
  range,
  onEnabledChange,
  onRangeChange,
}: SampleSetRangeSelectorProps) {
  const [useFullDataset, setUseFullDataset] = useState(range.limit === null);
  const [offset, setOffset] = useState(range.offset);
  const [limit, setLimit] = useState(range.limit || sampleSet.num_samples);

  // Update local state when props change
  useEffect(() => {
    setUseFullDataset(range.limit === null);
    setOffset(range.offset);
    setLimit(range.limit || sampleSet.num_samples);
  }, [range, sampleSet.num_samples]);

  // Calculate estimated size based on range
  // When selecting a partial dataset, we need to estimate proportionally
  // Note: Features and IQ samples are exported using the same OFFSET/LIMIT
  // So when you export features 100-200, you also get IQ samples 100-200
  const effectiveSamples = useFullDataset ? sampleSet.num_samples : (limit || sampleSet.num_samples);
  const totalSamples = sampleSet.num_samples;
  const proportion = totalSamples > 0 ? effectiveSamples / totalSamples : 0;
  
  // Calculate size: (features + IQ data) scaled by proportion
  const featureSize = effectiveSamples * (sampleSet.estimated_size_per_feature || 0);
  const iqSize = sampleSet.num_iq_samples * proportion * (sampleSet.estimated_size_per_iq || 0);
  const estimatedSize = Math.round(featureSize + iqSize);

  // Handle checkbox toggle
  const handleEnabledChange = (checked: boolean) => {
    onEnabledChange(checked);
    if (checked && !useFullDataset) {
      // When enabling with a range, ensure offset is valid
      onRangeChange(offset, limit);
    }
  };

  // Handle "Use Full Dataset" toggle
  const handleFullDatasetToggle = (checked: boolean) => {
    setUseFullDataset(checked);
    if (checked) {
      // Reset to full dataset
      setOffset(0);
      onRangeChange(0, null);
    } else {
      // Use current limit value
      onRangeChange(offset, limit);
    }
  };

  // Handle offset change
  const handleOffsetChange = (value: number) => {
    const clampedOffset = Math.max(0, Math.min(value, sampleSet.num_samples - 1));
    setOffset(clampedOffset);
    
    // Ensure limit doesn't exceed available samples
    const maxLimit = sampleSet.num_samples - clampedOffset;
    const newLimit = Math.min(limit, maxLimit);
    setLimit(newLimit);
    
    onRangeChange(clampedOffset, newLimit);
  };

  // Handle limit change
  const handleLimitChange = (value: number) => {
    const maxLimit = sampleSet.num_samples - offset;
    const clampedLimit = Math.max(1, Math.min(value, maxLimit));
    setLimit(clampedLimit);
    onRangeChange(offset, clampedLimit);
  };

  return (
    <div className="border rounded p-3 mb-2" style={{ backgroundColor: enabled ? '#f8f9fa' : '#fff' }}>
      {/* Header: Enable checkbox and dataset name */}
      <div className="form-check mb-2">
        <input
          className="form-check-input"
          type="checkbox"
          id={`sample-set-${sampleSet.id}`}
          checked={enabled}
          onChange={e => handleEnabledChange(e.target.checked)}
        />
        <label className="form-check-label fw-bold" htmlFor={`sample-set-${sampleSet.id}`}>
          {sampleSet.name}
        </label>
        <small className="text-muted ms-2">
          ({sampleSet.num_samples.toLocaleString()} samples, {formatBytes(sampleSet.estimated_size_bytes)})
        </small>
      </div>

      {/* Range selection (only show when enabled) */}
      {enabled && (
        <div className="ms-4">
          {/* Full Dataset Toggle */}
          <div className="form-check mb-2">
            <input
              className="form-check-input"
              type="checkbox"
              id={`full-dataset-${sampleSet.id}`}
              checked={useFullDataset}
              onChange={e => handleFullDatasetToggle(e.target.checked)}
            />
            <label className="form-check-label" htmlFor={`full-dataset-${sampleSet.id}`}>
              Export full dataset
            </label>
          </div>

          {/* Range controls (only show when not using full dataset) */}
          {!useFullDataset && (
            <div className="mt-2">
              {/* Offset input */}
              <div className="row mb-2">
                <div className="col-sm-4">
                  <label htmlFor={`offset-${sampleSet.id}`} className="form-label small mb-1">
                    Start Offset
                  </label>
                </div>
                <div className="col-sm-8">
                  <input
                    type="number"
                    className="form-control form-control-sm"
                    id={`offset-${sampleSet.id}`}
                    min={0}
                    max={sampleSet.num_samples - 1}
                    value={offset}
                    onChange={e => handleOffsetChange(parseInt(e.target.value) || 0)}
                  />
                </div>
              </div>

              {/* Limit input */}
              <div className="row mb-2">
                <div className="col-sm-4">
                  <label htmlFor={`limit-${sampleSet.id}`} className="form-label small mb-1">
                    Number of Samples
                  </label>
                </div>
                <div className="col-sm-8">
                  <input
                    type="number"
                    className="form-control form-control-sm"
                    id={`limit-${sampleSet.id}`}
                    min={1}
                    max={sampleSet.num_samples - offset}
                    value={limit}
                    onChange={e => handleLimitChange(parseInt(e.target.value) || 1)}
                  />
                </div>
              </div>

              {/* Range slider */}
              <div className="mb-2">
                <label className="form-label small mb-1">Range Preview</label>
                <div className="d-flex align-items-center gap-2">
                  <input
                    type="range"
                    className="form-range"
                    min={0}
                    max={sampleSet.num_samples}
                    value={offset}
                    onChange={e => handleOffsetChange(parseInt(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <small className="text-muted" style={{ minWidth: '120px' }}>
                    {offset.toLocaleString()} - {(offset + limit).toLocaleString()}
                  </small>
                </div>
              </div>

              {/* Size estimate */}
              <div className="alert alert-info py-1 px-2 mb-0">
                <small>
                  <strong>Estimated size:</strong> {formatBytes(estimatedSize)} ({effectiveSamples.toLocaleString()}{' '}
                  samples)
                </small>
              </div>
            </div>
          )}

          {/* Full dataset size estimate */}
          {useFullDataset && (
            <div className="alert alert-info py-1 px-2 mb-0 ms-4">
              <small>
                <strong>Estimated size:</strong> {formatBytes(estimatedSize)} ({effectiveSamples.toLocaleString()}{' '}
                samples)
              </small>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
