/**
 * GenerateDataDialog Component
 * 
 * Modal for generating synthetic RF localization datasets
 */

import React, { useState } from 'react';
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

  // Acceleration mode (AUTO, CPU, GPU)
  const [accelerationMode, setAccelerationMode] = useState<'auto' | 'cpu' | 'gpu'>('auto');

  // Advanced options toggle
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

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
    frequency_mhz: 144.0,  // 2m band (REQUIRED)
    tx_power_dbm: wattToDbm(2.0), // 2W = ~33dBm (REQUIRED)
    min_snr_db: 3.0,       // Reasonable minimum SNR (REQUIRED)
    min_receivers: 3,      // Minimum triangulation (REQUIRED)
    max_gdop: 150.0,       // Acceptable geometry (150 for clustered receivers)
    use_srtm_terrain: true, // Use SRTM terrain data (production baseline)
    use_random_receivers: false,
    seed: undefined,
    tx_antenna_dist: undefined,
    rx_antenna_dist: undefined,
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
      // Map acceleration mode to use_gpu value
      const use_gpu = accelerationMode === 'auto' ? null : accelerationMode === 'gpu' ? true : false;
      
      await generateSyntheticData({
        ...formData,
        use_gpu,
      });
      
      // Reset form to production baseline defaults
      setPowerValueWatt(2.0);
      setPowerUnit('watt');
      setAccelerationMode('auto');
      setShowAdvancedOptions(false);
      setFormData({
        name: '',
        description: '',
        dataset_type: 'feature_based',
        num_samples: 10000,
        frequency_mhz: 144.0,
        tx_power_dbm: wattToDbm(2.0),
        min_snr_db: 3.0,
        min_receivers: 3,
        max_gdop: 150.0,
        use_srtm_terrain: true,
        use_random_receivers: false,
        seed: undefined,
        tx_antenna_dist: undefined,
        rx_antenna_dist: undefined,
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

                  <div className="mb-3">
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
                    <div className="form-text">Default: 150 (recommended for clustered receivers)</div>
                    {formData.max_gdop < 120 && (
                      <div className="alert alert-warning d-flex align-items-start mt-2 mb-0" role="alert">
                        <i className="ph ph-warning me-2 mt-1"></i>
                        <div className="small">
                          <strong>Low Success Rate Expected:</strong> GDOP &lt;{formData.max_gdop} may result in high rejection rates (&lt;5% success) with clustered Italian WebSDRs. Consider increasing to ≥150 for better sample generation efficiency.
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Terrain Options */}
                <div className="mb-3">
                  <h6 className="fw-semibold mb-3">Terrain Options</h6>
                  
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="use_srtm_terrain"
                      checked={formData.use_srtm_terrain}
                      onChange={(e) => handleInputChange('use_srtm_terrain', e.target.checked)}
                      disabled={isLoading}
                    />
                    <label className="form-check-label" htmlFor="use_srtm_terrain">
                      Use SRTM Terrain Data
                    </label>
                    <div className="form-text">
                      Enable real terrain elevation (production baseline: ON)
                    </div>
                  </div>
                </div>

                {/* Processing Acceleration */}
                <div className="mb-4">
                  <h6 className="fw-semibold mb-3">Processing Acceleration</h6>
                  
                  <div className="btn-group w-100" role="group" aria-label="Acceleration mode selector">
                    <button
                      type="button"
                      className={`btn ${accelerationMode === 'auto' ? 'btn-primary' : 'btn-outline-primary'}`}
                      onClick={() => setAccelerationMode('auto')}
                      disabled={isLoading}
                    >
                      <i className="ph ph-magic-wand me-2"></i>
                      AUTO
                    </button>
                    <button
                      type="button"
                      className={`btn ${accelerationMode === 'cpu' ? 'btn-primary' : 'btn-outline-primary'}`}
                      onClick={() => setAccelerationMode('cpu')}
                      disabled={isLoading}
                    >
                      <i className="ph ph-cpu me-2"></i>
                      CPU
                    </button>
                    <button
                      type="button"
                      className={`btn ${accelerationMode === 'gpu' ? 'btn-primary' : 'btn-outline-primary'}`}
                      onClick={() => setAccelerationMode('gpu')}
                      disabled={isLoading}
                    >
                      <i className="ph ph-lightning me-2"></i>
                      GPU
                    </button>
                  </div>
                  
                  <div className="form-text mt-2">
                    {accelerationMode === 'auto' && 'Auto-detect GPU availability (recommended)'}
                    {accelerationMode === 'cpu' && 'Force CPU processing (slower but always available)'}
                    {accelerationMode === 'gpu' && 'Force GPU acceleration (requires CUDA-capable GPU)'}
                  </div>
                </div>

                {/* Random Receivers Toggle */}
                {formData.dataset_type === 'feature_based' && (
                  <div className="mb-3">
                    <div className="form-check form-switch">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="use_random_receivers"
                        checked={formData.use_random_receivers}
                        onChange={(e) => handleInputChange('use_random_receivers', e.target.checked)}
                        disabled={isLoading}
                      />
                      <label className="form-check-label" htmlFor="use_random_receivers">
                        Use Random Receiver Placement
                      </label>
                      <div className="form-text">
                        Generate random receiver positions instead of using fixed Italian WebSDRs
                      </div>
                    </div>
                  </div>
                )}

                {/* Advanced Options */}
                <div className="mb-4">
                  <button
                    type="button"
                    className="btn btn-link text-decoration-none p-0 d-flex align-items-center gap-2"
                    onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                    disabled={isLoading}
                  >
                    <i className={`ph ${showAdvancedOptions ? 'ph-caret-down' : 'ph-caret-right'}`}></i>
                    <span className="fw-semibold">Advanced Options</span>
                  </button>

                  {showAdvancedOptions && (
                    <div className="mt-3 border rounded p-3 bg-light text-dark">
                      {/* Random Seed */}
                      <div className="mb-3">
                        <label htmlFor="seed" className="form-label">
                          Random Seed
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="seed"
                          value={formData.seed ?? ''}
                          onChange={(e) => handleInputChange('seed', e.target.value ? parseInt(e.target.value) : undefined)}
                          placeholder="Leave empty for random"
                          disabled={isLoading}
                        />
                        <div className="form-text">
                          Set a fixed seed for reproducible datasets (optional)
                        </div>
                      </div>

                      {/* TX Antenna Distribution */}
                      <div className="mb-3">
                        <label className="form-label fw-semibold">
                          TX Antenna Distribution
                        </label>
                        <div className="form-text mb-2">
                          Probability distribution of transmitter antenna types (must sum to 1.0)
                        </div>
                        
                        <div className="row g-2">
                          <div className="col-md-4">
                            <label htmlFor="tx_whip" className="form-label small">Whip</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="tx_whip"
                              value={formData.tx_antenna_dist?.whip ?? 0.60}
                              onChange={(e) => handleInputChange('tx_antenna_dist', {
                                ...formData.tx_antenna_dist,
                                whip: parseFloat(e.target.value) || 0.60
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                          <div className="col-md-4">
                            <label htmlFor="tx_rubber_duck" className="form-label small">Rubber Duck</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="tx_rubber_duck"
                              value={formData.tx_antenna_dist?.rubber_duck ?? 0.38}
                              onChange={(e) => handleInputChange('tx_antenna_dist', {
                                ...formData.tx_antenna_dist,
                                rubber_duck: parseFloat(e.target.value) || 0.38
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                          <div className="col-md-4">
                            <label htmlFor="tx_portable_directional" className="form-label small">Portable Directional</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="tx_portable_directional"
                              value={formData.tx_antenna_dist?.portable_directional ?? 0.02}
                              onChange={(e) => handleInputChange('tx_antenna_dist', {
                                ...formData.tx_antenna_dist,
                                portable_directional: parseFloat(e.target.value) || 0.02
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                        </div>
                        {formData.tx_antenna_dist && (
                          <div className="form-text mt-1">
                            Sum: {((formData.tx_antenna_dist.whip ?? 0.60) + 
                                   (formData.tx_antenna_dist.rubber_duck ?? 0.38) + 
                                   (formData.tx_antenna_dist.portable_directional ?? 0.02)).toFixed(2)}
                            {Math.abs(((formData.tx_antenna_dist.whip ?? 0.60) + 
                                      (formData.tx_antenna_dist.rubber_duck ?? 0.38) + 
                                      (formData.tx_antenna_dist.portable_directional ?? 0.02)) - 1.0) > 0.01 && (
                              <span className="text-danger ms-2">⚠ Must sum to 1.0</span>
                            )}
                          </div>
                        )}
                      </div>

                      {/* RX Antenna Distribution */}
                      <div className="mb-0">
                        <label className="form-label fw-semibold">
                          RX Antenna Distribution
                        </label>
                        <div className="form-text mb-2">
                          Probability distribution of receiver antenna types (must sum to 1.0)
                        </div>
                        
                        <div className="row g-2">
                          <div className="col-md-4">
                            <label htmlFor="rx_omni_vertical" className="form-label small">Omni Vertical</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="rx_omni_vertical"
                              value={formData.rx_antenna_dist?.omni_vertical ?? 0.80}
                              onChange={(e) => handleInputChange('rx_antenna_dist', {
                                ...formData.rx_antenna_dist,
                                omni_vertical: parseFloat(e.target.value) || 0.80
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                          <div className="col-md-4">
                            <label htmlFor="rx_yagi" className="form-label small">Yagi</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="rx_yagi"
                              value={formData.rx_antenna_dist?.yagi ?? 0.15}
                              onChange={(e) => handleInputChange('rx_antenna_dist', {
                                ...formData.rx_antenna_dist,
                                yagi: parseFloat(e.target.value) || 0.15
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                          <div className="col-md-4">
                            <label htmlFor="rx_collinear" className="form-label small">Collinear</label>
                            <input
                              type="number"
                              className="form-control form-control-sm"
                              id="rx_collinear"
                              value={formData.rx_antenna_dist?.collinear ?? 0.05}
                              onChange={(e) => handleInputChange('rx_antenna_dist', {
                                ...formData.rx_antenna_dist,
                                collinear: parseFloat(e.target.value) || 0.05
                              })}
                              min={0}
                              max={1}
                              step={0.01}
                              disabled={isLoading}
                            />
                          </div>
                        </div>
                        {formData.rx_antenna_dist && (
                          <div className="form-text mt-1">
                            Sum: {((formData.rx_antenna_dist.omni_vertical ?? 0.80) + 
                                   (formData.rx_antenna_dist.yagi ?? 0.15) + 
                                   (formData.rx_antenna_dist.collinear ?? 0.05)).toFixed(2)}
                            {Math.abs(((formData.rx_antenna_dist.omni_vertical ?? 0.80) + 
                                      (formData.rx_antenna_dist.yagi ?? 0.15) + 
                                      (formData.rx_antenna_dist.collinear ?? 0.05)) - 1.0) > 0.01 && (
                              <span className="text-danger ms-2">⚠ Must sum to 1.0</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Info Box */}
                <div className="alert alert-info d-flex align-items-start" role="alert">
                  <i className="ph ph-info me-2 mt-1"></i>
                  <div>
                    <strong>Configuration Summary</strong>
                    <ul className="mb-0 mt-2 small">
                      <li><strong>Dataset Type:</strong> {formData.dataset_type === 'iq_raw' ? 'IQ Raw (CNN training)' : 'Feature-Based (traditional)'}</li>
                      <li><strong>Samples:</strong> {formData.num_samples?.toLocaleString()} samples</li>
                      <li><strong>RF:</strong> {formData.frequency_mhz} MHz, {powerValueWatt.toFixed(1)}W TX power</li>
                      <li><strong>Quality:</strong> Min {formData.min_receivers} receivers, SNR ≥{formData.min_snr_db} dB, GDOP ≤{formData.max_gdop}</li>
                      <li><strong>Terrain:</strong> {formData.use_srtm_terrain ? 'SRTM enabled' : 'Flat earth model'}</li>
                      <li><strong>Acceleration:</strong> {accelerationMode.toUpperCase()} mode</li>
                      {formData.seed !== undefined && (
                        <li><strong>Seed:</strong> {formData.seed} (reproducible)</li>
                      )}
                      {formData.tx_antenna_dist && (
                        <li><strong>TX Antennas:</strong> Whip {(formData.tx_antenna_dist.whip * 100).toFixed(0)}%, Rubber Duck {(formData.tx_antenna_dist.rubber_duck * 100).toFixed(0)}%, Directional {(formData.tx_antenna_dist.portable_directional * 100).toFixed(0)}%</li>
                      )}
                      {formData.rx_antenna_dist && (
                        <li><strong>RX Antennas:</strong> Omni {(formData.rx_antenna_dist.omni_vertical * 100).toFixed(0)}%, Yagi {(formData.rx_antenna_dist.yagi * 100).toFixed(0)}%, Collinear {(formData.rx_antenna_dist.collinear * 100).toFixed(0)}%</li>
                      )}
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
