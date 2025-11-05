/**
 * SampleExplorerTab Component
 * 
 * Tab view that combines the map and details panel for exploring a synthetic sample
 * - Left side: Interactive map with TX/RX positions
 * - Right side: Sample details with receiver table and propagation breakdown
 */

import React, { useState } from 'react';
import { Row, Col } from 'react-bootstrap';
import type { SyntheticSample } from '../../types';
import { SampleMapView } from './SampleMapView';
import { SampleDetailsPanel } from './SampleDetailsPanel';

interface SampleExplorerTabProps {
    sample: SyntheticSample;
}

export const SampleExplorerTab: React.FC<SampleExplorerTabProps> = ({ sample }) => {
    const [selectedRxId, setSelectedRxId] = useState<string | null>(null);

    return (
        <Row>
            <Col md={7}>
                <SampleMapView
                    sample={sample}
                    selectedRxId={selectedRxId}
                    onRxSelect={setSelectedRxId}
                    height={600}
                />
            </Col>
            <Col md={5}>
                <SampleDetailsPanel
                    sample={sample}
                    selectedRxId={selectedRxId}
                    onRxSelect={setSelectedRxId}
                />
            </Col>
        </Row>
    );
};
