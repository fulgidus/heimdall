/**
 * GenerateDataDialog Component
 * 
 * Modal for generating synthetic RF localization datasets
 */

import React, { useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import type { SyntheticDataRequest } from '../../types';

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
    num_samples: 10000,
    inside_ratio: 0.75,    // Production baseline
    frequency_mhz: 144.0,  // 2m band
    tx_power_dbm: wattToDbm(2.0), // 2W = ~33dBm
    min_snr_db: 3.0,       // Reasonable minimum SNR
    min_receivers: 3,      // Minimum triangulation
    max_gdop: 100.0,       // Acceptable geometry
    use_srtm: true,        // Use SRTM terrain data (production baseline)
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
        num_samples: 10000,
        inside_ratio: 0.75,
        frequency_mhz: 144.0,
        tx_power_dbm: wattToDbm(2.0),
        min_snr_db: 3.0,
        min_receivers: 3,
        max_gdop: 100.0,
        use_srtm: true,
      });
      
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate dataset');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
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

                {/* Info Box */}
                <div className="alert alert-info d-flex align-items-start" role="alert">
                  <i className="ph ph-info me-2 mt-1"></i>
                  <div>
                    <strong>Production Baseline Configuration</strong>
                    <ul className="mb-0 mt-2 small">
                      <li>10,000 samples with 75% inside coverage</li>
                      <li>144 MHz (2m band), 2W TX power (~33 dBm)</li>
                      <li>Min 3 receivers, SNR ≥3 dB, GDOP ≤100</li>
                      <li>SRTM terrain enabled for realistic propagation</li>
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
    </>
  );
};
