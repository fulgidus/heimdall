/**
 * GenerateDataDialog Component
 * 
 * Modal for generating synthetic RF localization datasets
 */

import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useTrainingStore } from '../../../../store/trainingStore';
import type { SyntheticDataRequest } from '../../types';
import { usePortal } from '@/hooks/usePortal';

interface GenerateDataDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export const GenerateDataDialog: React.FC<GenerateDataDialogProps> = ({
  isOpen,
  onClose,
}) => {
  const { generateSyntheticData } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(isOpen);

  // Power unit toggle (Watt or dBm)
  const [powerUnit, setPowerUnit] = useState<'watt' | 'dbm'>('watt');
  const [powerValueWatt, setPowerValueWatt] = useState(2.0); // Default 2W = ~33dBm

  // Helper functions for power conversion
  const wattToDbm = (watt: number): number => {
    return 10 * Math.log10(watt * 1000);
  };

  const dbmToWatt = (dbm: number): number => {
    return Math.pow(10, dbm / 10) / 1000;
  };

  // Production Baseline defaults (balanced quality/performance with SRTM terrain)
  const [formData, setFormData] = useState<SyntheticDataRequest>({
    name: '',
    description: '',
    dataset_type: 'feature_based',  // Default: traditional feature-based
    num_samples: 10000,
    inside_ratio: 0.75,    // Production baseline
    frequency_mhz: 144.0,  // 2m band
    tx_power_dbm: wattToDbm(2.0), // 2W = ~33dBm
    min_snr_db: 3.0,       // Reasonable minimum SNR
    min_receivers: 3,      // Minimum triangulation
    max_gdop: 100.0,       // Acceptable geometry
    use_srtm: true,        // Use SRTM terrain data (production baseline)
    // Random receiver defaults (for iq_raw datasets)
    use_random_receivers: false,
    min_receivers_count: 5,
    max_receivers_count: 10,
    receiver_seed: undefined,
    area_lat_min: 44.0,
    area_lat_max: 46.0,
    area_lon_min: 7.0,
    area_lon_max: 10.0,
  });

  const handleInputChange = (field: keyof SyntheticDataRequest, value: string | number | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePowerChange = (value: number) => {
    if (powerUnit === 'watt') {
      setPowerValueWatt(value);
      setFormData(prev => ({ ...prev, tx_power_dbm: wattToDbm(value) }));
    } else {
      const watt = dbmToWatt(value);
      setPowerValueWatt(watt);
      setFormData(prev => ({ ...prev, tx_power_dbm: value }));
    }
  };

  const togglePowerUnit = () => {
    setPowerUnit(prev => prev === 'watt' ? 'dbm' : 'watt');
  };



  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('Dataset name is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      await generateSyntheticData(formData);
      
      // Reset form to production baseline defaults
      setPowerValueWatt(2.0);
      setPowerUnit('watt');
      setFormData({
        name: '',
        description: '',
        dataset_type: 'feature_based',
        num_samples: 10000,
        inside_ratio: 0.75,
        frequency_mhz: 144.0,
        tx_power_dbm: wattToDbm(2.0),
        min_snr_db: 3.0,
        min_receivers: 3,
        max_gdop: 100.0,
        use_srtm: true,
        use_random_receivers: false,
        min_receivers_count: 5,
        max_receivers_count: 10,
        receiver_seed: undefined,
        area_lat_min: 44.0,
        area_lat_max: 46.0,
        area_lon_min: 7.0,
        area_lon_max: 10.0,
      });
      
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate dataset');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen || !portalTarget) return null;

  return createPortal(
    <>
      {/* Bootstrap Modal Backdrop */}
      <div 
        className="modal-backdrop fade show"
        onClick={onClose}
      ></div>

      {/* Bootstrap Modal */}
      <div 
        className="modal fade show d-block" 
        tabIndex={-1}
        style={{ display: 'block' }}
      >
        <div className="modal-dialog modal-lg modal-dialog-scrollable">
          <div className="modal-content">
            <form onSubmit={handleSubmit}>
              {/* Modal Header */}
              <div className="modal-header">
                <h5 className="modal-title">
                  <i className="ph ph-database me-2"></i>
                  Generate Synthetic Dataset
                </h5>
                <button
                  type="button"
                  className="btn-close"
                  onClick={onClose}
                  aria-label="Close"
                  disabled={isLoading}
                ></button>
              </div>

              {/* Modal Body */}
              <div className="modal-body">
                {/* Error Alert */}
                {error && (
                  <div className="alert alert-danger" role="alert">
                    <i className="ph ph-warning-circle me-2"></i>
                    {error}
                  </div>
                )}

                {/* Dataset Info */}
                <div className="mb-4">
                  <h6 className="fw-semibold mb-3">Dataset Information</h6>
                  
                  <div className="mb-3">
                    <label htmlFor="name" className="form-label">
                      Dataset Name <span className="text-danger">*</span>
                    </label>
                    <input
                      type="text"
                      className="form-control"
                      id="name"
                      value={formData.name}
                      onChange={(e) => handleInputChange('name', e.target.value)}
                      placeholder="e.g., baseline_10k_75pct"
                      required
                      disabled={isLoading}
                    />
                  </div>

                  <div className="mb-3">
                    <label htmlFor="description" className="form-label">
                      Description
                    </label>
                    <textarea
                      className="form-control"
                      id="description"
                      rows={2}
                      value={formData.description}
                      onChange={(e) => handleInputChange('description', e.target.value)}
                      placeholder="Optional description of the dataset"
                      disabled={isLoading}
                    />
                  </div>

                  {/* Dataset Type Selector */}
                  <div className="mb-3">
                    <label className="form-label">
                      Dataset Type <span className="text-danger">*</span>
                    </label>
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="radio"
                        name="dataset_type"
                        id="dataset_type_feature"
                        value="feature_based"
                        checked={formData.dataset_type === 'feature_based'}
                        onChange={(e) => handleInputChange('dataset_type', e.target.value)}
                        disabled={isLoading}
                      />
                      <label className="form-check-label" htmlFor="dataset_type_feature">
                        <strong>Feature-Based</strong> (TriangulationModel)
                        <div className="form-text">
                          Uses 7 fixed Italian WebSDRs. Stores extracted features only (mel-spectrograms, MFCCs).
                          Smaller storage footprint (~1-5 MB per 1000 samples).
                        </div>
                      </label>
                    </div>
                    <div className="form-check mt-2">
                      <input
                        className="form-check-input"
                        type="radio"
                        name="dataset_type"
                        id="dataset_type_iq"
                        value="iq_raw"
                        checked={formData.dataset_type === 'iq_raw'}
                        onChange={(e) => {
                          handleInputChange('dataset_type', e.target.value);
                          // Automatically enable random receivers for iq_raw
                          handleInputChange('use_random_receivers', true);
                        }}
                        disabled={isLoading}
                      />
                      <label className="form-check-label" htmlFor="dataset_type_iq">
                        <strong>IQ Raw</strong> (LocalizationNet/CNN)
                        <div className="form-text">
                          Uses 5-10 random receivers per sample. Stores ALL raw IQ samples + extracted features.
                          Larger storage (~50-200 MB per 1000 samples). Required for CNN training.
                        </div>
                      </label>
                    </div>
                  </div>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="num_samples" className="form-label">
                        Number of Samples
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="num_samples"
                        value={formData.num_samples}
                        onChange={(e) => handleInputChange('num_samples', parseInt(e.target.value))}
                        min={100}
                        max={1000000}
                        disabled={isLoading}
                      />
                      <div className="form-text">Default: 10,000 (production baseline)</div>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="inside_ratio" className="form-label">
                        Inside Coverage Ratio
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="inside_ratio"
                        value={formData.inside_ratio}
                        onChange={(e) => handleInputChange('inside_ratio', parseFloat(e.target.value))}
                        min={0}
                        max={1}
                        step={0.05}
                        disabled={isLoading}
                      />
                      <div className="form-text">0-1, Default: 0.75 (75% inside)</div>
                    </div>
                  </div>
                </div>

                {/* RF Parameters */}
                <div className="mb-4">
                  <h6 className="fw-semibold mb-3">RF Parameters</h6>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="frequency_mhz" className="form-label">
                        Frequency (MHz)
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="frequency_mhz"
                        value={formData.frequency_mhz}
                        onChange={(e) => handleInputChange('frequency_mhz', parseFloat(e.target.value))}
                        min={1}
                        max={10000}
                        step={0.1}
                        disabled={isLoading}
                      />
                      <div className="form-text">Default: 144.0 MHz (2m band)</div>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label className="form-label">
                        TX Power
                      </label>
                      <div className="input-group">
                        <input
                          type="number"
                          className="form-control"
                          value={powerUnit === 'watt' ? powerValueWatt.toFixed(2) : formData.tx_power_dbm?.toFixed(2)}
                          onChange={(e) => handlePowerChange(parseFloat(e.target.value))}
                          step={powerUnit === 'watt' ? 0.1 : 1}
                          disabled={isLoading}
                        />
                        <button
                          type="button"
                          className="btn btn-outline-secondary"
                          onClick={togglePowerUnit}
                          disabled={isLoading}
                        >
                          {powerUnit === 'watt' ? 'W' : 'dBm'}
                        </button>
                      </div>
                      <div className="form-text">
                        {powerUnit === 'watt' 
                          ? `${wattToDbm(powerValueWatt).toFixed(1)} dBm` 
                          : `${powerValueWatt.toFixed(2)} W`}
                      </div>
                    </div>
                  </div>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="min_snr_db" className="form-label">
                        Minimum SNR (dB)
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="min_snr_db"
                        value={formData.min_snr_db}
                        onChange={(e) => handleInputChange('min_snr_db', parseFloat(e.target.value))}
                        min={0}
                        max={50}
                        step={0.5}
                        disabled={isLoading}
                      />
                      <div className="form-text">Default: 3.0 dB</div>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="min_receivers" className="form-label">
                        Minimum Receivers
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="min_receivers"
                        value={formData.min_receivers}
                        onChange={(e) => handleInputChange('min_receivers', parseInt(e.target.value))}
                        min={3}
                        max={7}
                        disabled={isLoading}
                      />
                      <div className="form-text">Default: 3 (triangulation minimum)</div>
                    </div>
                  </div>

                  <div className="mb-3">
                    <label htmlFor="max_gdop" className="form-label">
                      Maximum GDOP
                    </label>
                    <input
                      type="number"
                      className="form-control"
                      id="max_gdop"
                      value={formData.max_gdop}
                      onChange={(e) => handleInputChange('max_gdop', parseFloat(e.target.value))}
                      min={1}
                      max={1000}
                      step={1}
                      disabled={isLoading}
                    />
                    <div className="form-text">Default: 100 (acceptable geometry)</div>
                  </div>
                </div>

                {/* Terrain Options */}
                <div className="mb-3">
                  <h6 className="fw-semibold mb-3">Terrain Options</h6>
                  
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="use_srtm"
                      checked={formData.use_srtm}
                      onChange={(e) => handleInputChange('use_srtm', e.target.checked)}
                      disabled={isLoading}
                    />
                    <label className="form-check-label" htmlFor="use_srtm">
                      Use SRTM Terrain Data
                    </label>
                    <div className="form-text">
                      Enable real terrain elevation (production baseline: ON)
                    </div>
                  </div>
                </div>

                {/* Random Receivers Configuration (conditional) */}
                {(formData.dataset_type === 'iq_raw' || formData.use_random_receivers) && (
                  <div className="mb-4">
                    <h6 className="fw-semibold mb-3">
                      Random Receivers Configuration
                      {formData.dataset_type === 'iq_raw' && (
                        <span className="badge bg-primary ms-2">Required for IQ Raw</span>
                      )}
                    </h6>
                    
                    <div className="alert alert-info d-flex align-items-start mb-3" role="alert">
                      <i className="ph ph-info me-2 mt-1"></i>
                      <div className="small">
                        Each sample will use a random subset of receivers (between min and max count)
                        placed within the specified geographic area. This creates dataset diversity
                        essential for CNN generalization.
                      </div>
                    </div>

                    <div className="row">
                      <div className="col-md-6 mb-3">
                        <label htmlFor="min_receivers_count" className="form-label">
                          Min Receivers per Sample
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="min_receivers_count"
                          value={formData.min_receivers_count}
                          onChange={(e) => handleInputChange('min_receivers_count', parseInt(e.target.value))}
                          min={3}
                          max={15}
                          disabled={isLoading}
                        />
                        <div className="form-text">Default: 5 (minimum 3 for triangulation)</div>
                      </div>

                      <div className="col-md-6 mb-3">
                        <label htmlFor="max_receivers_count" className="form-label">
                          Max Receivers per Sample
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="max_receivers_count"
                          value={formData.max_receivers_count}
                          onChange={(e) => handleInputChange('max_receivers_count', parseInt(e.target.value))}
                          min={3}
                          max={15}
                          disabled={isLoading}
                        />
                        <div className="form-text">Default: 10</div>
                      </div>
                    </div>

                    <div className="mb-3">
                      <label className="form-label">Geographic Area (Receiver Placement)</label>
                      <div className="row">
                        <div className="col-md-6 mb-2">
                          <label htmlFor="area_lat_min" className="form-label small">
                            Latitude Min (°)
                          </label>
                          <input
                            type="number"
                            className="form-control form-control-sm"
                            id="area_lat_min"
                            value={formData.area_lat_min}
                            onChange={(e) => handleInputChange('area_lat_min', parseFloat(e.target.value))}
                            min={-90}
                            max={90}
                            step={0.1}
                            disabled={isLoading}
                          />
                        </div>
                        <div className="col-md-6 mb-2">
                          <label htmlFor="area_lat_max" className="form-label small">
                            Latitude Max (°)
                          </label>
                          <input
                            type="number"
                            className="form-control form-control-sm"
                            id="area_lat_max"
                            value={formData.area_lat_max}
                            onChange={(e) => handleInputChange('area_lat_max', parseFloat(e.target.value))}
                            min={-90}
                            max={90}
                            step={0.1}
                            disabled={isLoading}
                          />
                        </div>
                      </div>
                      <div className="row">
                        <div className="col-md-6 mb-2">
                          <label htmlFor="area_lon_min" className="form-label small">
                            Longitude Min (°)
                          </label>
                          <input
                            type="number"
                            className="form-control form-control-sm"
                            id="area_lon_min"
                            value={formData.area_lon_min}
                            onChange={(e) => handleInputChange('area_lon_min', parseFloat(e.target.value))}
                            min={-180}
                            max={180}
                            step={0.1}
                            disabled={isLoading}
                          />
                        </div>
                        <div className="col-md-6 mb-2">
                          <label htmlFor="area_lon_max" className="form-label small">
                            Longitude Max (°)
                          </label>
                          <input
                            type="number"
                            className="form-control form-control-sm"
                            id="area_lon_max"
                            value={formData.area_lon_max}
                            onChange={(e) => handleInputChange('area_lon_max', parseFloat(e.target.value))}
                            min={-180}
                            max={180}
                            step={0.1}
                            disabled={isLoading}
                          />
                        </div>
                      </div>
                      <div className="form-text">
                        Default: Northern Italy (Lat: 44-46°, Lon: 7-10°)
                      </div>
                    </div>

                    <div className="mb-3">
                      <label htmlFor="receiver_seed" className="form-label">
                        Random Seed (Optional)
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="receiver_seed"
                        value={formData.receiver_seed || ''}
                        onChange={(e) => handleInputChange('receiver_seed', e.target.value ? parseInt(e.target.value) : undefined)}
                        placeholder="Leave empty for random generation"
                        disabled={isLoading}
                      />
                      <div className="form-text">
                        Set a seed for reproducible receiver placement (useful for experiments)
                      </div>
                    </div>
                  </div>
                )}

                {/* Info Box */}
                <div className="alert alert-info d-flex align-items-start" role="alert">
                  <i className="ph ph-info me-2 mt-1"></i>
                  <div>
                    <strong>Configuration Summary</strong>
                    <ul className="mb-0 mt-2 small">
                      <li><strong>Dataset Type:</strong> {formData.dataset_type === 'iq_raw' ? 'IQ Raw (CNN training)' : 'Feature-Based (traditional)'}</li>
                      <li><strong>Samples:</strong> {formData.num_samples?.toLocaleString()} with {((formData.inside_ratio || 0) * 100).toFixed(0)}% inside coverage</li>
                      <li><strong>RF:</strong> {formData.frequency_mhz} MHz, {powerValueWatt.toFixed(1)}W TX power</li>
                      <li><strong>Quality:</strong> Min {formData.min_receivers} receivers, SNR ≥{formData.min_snr_db} dB, GDOP ≤{formData.max_gdop}</li>
                      {formData.dataset_type === 'iq_raw' && (
                        <li><strong>Receivers:</strong> {formData.min_receivers_count}-{formData.max_receivers_count} random per sample</li>
                      )}
                      <li><strong>Terrain:</strong> {formData.use_srtm ? 'SRTM enabled' : 'Flat earth model'}</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={onClose}
                  disabled={isLoading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary d-flex align-items-center gap-2"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                      Generating...
                    </>
                  ) : (
                    <>
                      <i className="ph ph-play-circle"></i>
                      Generate Dataset
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>,
    portalTarget
  );
};
