/**
 * Training Page
 * 
 * Main page for managing ML model training, monitoring metrics, and model export/import
 */

import React, { useState } from 'react';
import { JobsTab } from './components/JobsTab/JobsTab';
import { MetricsTab } from './components/MetricsTab/MetricsTab';
import { ModelsTab } from './components/ModelsTab/ModelsTab';
import { SyntheticTab } from './components/SyntheticTab/SyntheticTab';

type TabId = 'jobs' | 'metrics' | 'models' | 'synthetic';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

const tabs: Tab[] = [
  {
    id: 'jobs',
    label: 'Training Jobs',
    icon: 'ph-clipboard-text',
  },
  {
    id: 'metrics',
    label: 'Metrics',
    icon: 'ph-chart-bar',
  },
  {
    id: 'models',
    label: 'Trained Models',
    icon: 'ph-brain',
  },
  {
    id: 'synthetic',
    label: 'Synthetic Generation',
    icon: 'ph-database',
  },
];

export const Training: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('jobs');

  return (
    <>
      {/* Breadcrumb */}
      <nav className="page-header" aria-label="Breadcrumb">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ol className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/">Home</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  ML Training
                </li>
              </ol>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h1 className="mb-0">ML Training</h1>
                <p className="text-muted mt-1">
                  Train, monitor, and manage localization models
                </p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Tabs Card */}
      <div className="row">
        <div className="col-12">
          <div className="card">
            {/* Tab Navigation */}
            <div className="card-header border-bottom-0 pb-0">
              <ul className="nav nav-tabs profile-tabs" role="tablist">
                {tabs.map((tab) => (
                  <li className="nav-item" key={tab.id} role="presentation">
                    <button
                      className={`nav-link d-flex align-items-center gap-2 ${activeTab === tab.id ? 'active' : ''}`}
                      type="button"
                      role="tab"
                      aria-selected={activeTab === tab.id}
                      onClick={() => setActiveTab(tab.id)}
                    >
                      <i className={`ph ${tab.icon}`}></i>
                      <span>{tab.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {/* Tab Content */}
            <div className="card-body">
              <div className="tab-content">
                {activeTab === 'jobs' && <JobsTab />}
                {activeTab === 'metrics' && <MetricsTab />}
                {activeTab === 'models' && <ModelsTab />}
                {activeTab === 'synthetic' && <SyntheticTab />}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Training;
