/**
 * Spectrogram Viewer Component
 * Displays spectrogram visualization for RF signal data
 * 
 * Note: This is a placeholder implementation for Phase 7
 * Full spectrogram generation will be implemented in Phase 8 when backend provides data
 */

import React from 'react';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { Button } from './ui/button';

interface SpectrogramViewerProps {
    sessionId?: number;
    websdrId?: number;
    websdrName?: string;
    snr?: number;
    className?: string;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
    sessionId,
    websdrId,
    websdrName = 'WebSDR',
    snr,
    className = '',
}) => {
    // Placeholder: In Phase 8, this will fetch actual spectrogram from backend
    // const spectrogramUrl = `/api/v1/sessions/${sessionId}/spectrogram/${websdrId}`;

    const getSNRColor = (snrValue?: number) => {
        if (!snrValue) return 'text-slate-400';
        if (snrValue > 20) return 'text-green-400'; // Good
        if (snrValue >= 10) return 'text-yellow-400'; // Marginal
        return 'text-red-400'; // Poor
    };

    const getSNRLabel = (snrValue?: number) => {
        if (!snrValue) return 'N/A';
        if (snrValue > 20) return 'Good';
        if (snrValue >= 10) return 'Marginal';
        return 'Poor';
    };

    return (
        <div className={`bg-slate-800 border border-slate-700 rounded-lg overflow-hidden ${className}`}>
            {/* Header */}
            <div className="px-3 py-2 bg-slate-900 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <h4 className="text-sm font-semibold text-white">{websdrName}</h4>
                    {snr !== undefined && (
                        <span className={`text-xs font-medium ${getSNRColor(snr)}`}>
                            SNR: {snr.toFixed(1)} dB ({getSNRLabel(snr)})
                        </span>
                    )}
                </div>
                
                {/* Zoom controls (placeholder for Phase 8) */}
                <div className="flex gap-1">
                    <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-slate-400 hover:text-white"
                        title="Zoom in"
                        disabled
                    >
                        <ZoomIn className="w-3 h-3" />
                    </Button>
                    <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-slate-400 hover:text-white"
                        title="Zoom out"
                        disabled
                    >
                        <ZoomOut className="w-3 h-3" />
                    </Button>
                    <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-slate-400 hover:text-white"
                        title="Fullscreen"
                        disabled
                    >
                        <Maximize2 className="w-3 h-3" />
                    </Button>
                </div>
            </div>

            {/* Spectrogram Display - Placeholder */}
            <div className="aspect-video bg-slate-950 flex items-center justify-center relative">
                {/* Placeholder grid pattern */}
                <div className="absolute inset-0 opacity-10"
                    style={{
                        backgroundImage: `
                            linear-gradient(to right, #475569 1px, transparent 1px),
                            linear-gradient(to bottom, #475569 1px, transparent 1px)
                        `,
                        backgroundSize: '20px 20px'
                    }}
                />
                
                {/* Placeholder text */}
                <div className="text-center z-10">
                    <p className="text-slate-500 text-sm mb-1">Spectrogram Preview</p>
                    <p className="text-slate-600 text-xs">
                        {sessionId ? `Session ${sessionId} - WebSDR ${websdrId}` : 'No data available'}
                    </p>
                    <p className="text-slate-700 text-xs mt-2">
                        (Full implementation in Phase 8)
                    </p>
                </div>

                {/* Frequency/Time axis labels (placeholder) */}
                <div className="absolute bottom-2 left-2 text-xs text-slate-600">
                    Time →
                </div>
                <div className="absolute top-2 left-2 text-xs text-slate-600 transform -rotate-90 origin-left">
                    Frequency ↑
                </div>
            </div>
        </div>
    );
};

export default SpectrogramViewer;
