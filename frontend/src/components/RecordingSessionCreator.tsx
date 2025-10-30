'use client';

import React, { useState } from 'react';
import {
    Zap,
    AlertCircle,
    Radio,
    Clock,
    Waves,
} from 'lucide-react';
import { useSessionStore } from '../store/sessionStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface RecordingSessionCreatorProps {
    onSessionCreated?: (sessionId: string) => void;
}

export const RecordingSessionCreator: React.FC<RecordingSessionCreatorProps> = ({
    onSessionCreated,
}) => {
    // Form state
    const [sessionName, setSessionName] = useState('Session ' + new Date().toLocaleTimeString());
    const [frequency, setFrequency] = useState(145.5);
    const [duration, setDuration] = useState(30);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Store
    const { createSession, error, clearError } = useSessionStore();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);
        clearError();

        try {
            // TODO: This component is simplified and missing proper source selection.
            // The full implementation is in RecordingSession.tsx. This component should either:
            // 1. Add a source selection dropdown, or
            // 2. Be deprecated in favor of RecordingSession.tsx
            const newSession = await createSession({
                known_source_id: '00000000-0000-0000-0000-000000000000', // Placeholder - API will reject this
                session_name: sessionName,
                frequency_hz: Math.round(frequency * 1e6), // Convert MHz to Hz
                duration_seconds: duration,
            });

            // Reset form
            setSessionName('Session ' + new Date().toLocaleTimeString());
            setFrequency(145.5);
            setDuration(30);

            // Callback
            if (onSessionCreated) {
                onSessionCreated(newSession.id);
            }
        } catch (err) {
            // Error is already in store
            console.error('Failed to create session:', err);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <Card className="bg-slate-900 border-slate-800 w-full">
            <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                    <Radio className="w-5 h-5 text-purple-500" />
                    Create Recording Session
                </CardTitle>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Error Alert */}
                    {error && (
                        <Alert className="bg-red-900/20 border-red-800">
                            <AlertCircle className="h-4 w-4 text-red-500" />
                            <AlertDescription className="text-red-300">{error}</AlertDescription>
                        </Alert>
                    )}

                    {/* Session Name */}
                    <div className="space-y-2">
                        <Label htmlFor="session-name" className="text-slate-300 font-semibold">
                            Session Name
                        </Label>
                        <Input
                            id="session-name"
                            type="text"
                            placeholder="Enter session name"
                            value={sessionName}
                            onChange={(e) => setSessionName(e.target.value)}
                            disabled={isSubmitting}
                            className="bg-slate-800 border-slate-700 text-white placeholder-slate-500"
                        />
                        <p className="text-xs text-slate-400">
                            Friendly name to identify this recording session
                        </p>
                    </div>

                    {/* Frequency */}
                    <div className="space-y-2">
                        <Label htmlFor="frequency" className="text-slate-300 font-semibold flex items-center gap-2">
                            <Waves className="w-4 h-4 text-cyan-500" />
                            Frequency (MHz)
                        </Label>
                        <div className="flex gap-2">
                            <Input
                                id="frequency"
                                type="number"
                                step="0.001"
                                min="100"
                                max="1000"
                                value={frequency}
                                onChange={(e) => setFrequency(parseFloat(e.target.value))}
                                disabled={isSubmitting}
                                className="bg-slate-800 border-slate-700 text-white"
                            />
                            <div className="flex items-center px-4 bg-slate-800 border border-slate-700 rounded text-slate-400">
                                MHz
                            </div>
                        </div>
                        <p className="text-xs text-slate-400">
                            Common values: 145.500 (2m), 433.000 (70cm)
                        </p>
                    </div>

                    {/* Duration */}
                    <div className="space-y-2">
                        <Label htmlFor="duration" className="text-slate-300 font-semibold flex items-center gap-2">
                            <Clock className="w-4 h-4 text-green-500" />
                            Duration (seconds)
                        </Label>
                        <div className="flex gap-2">
                            <Input
                                id="duration"
                                type="number"
                                step="1"
                                min="5"
                                max="300"
                                value={duration}
                                onChange={(e) => setDuration(parseInt(e.target.value))}
                                disabled={isSubmitting}
                                className="bg-slate-800 border-slate-700 text-white"
                            />
                            <div className="flex items-center px-4 bg-slate-800 border border-slate-700 rounded text-slate-400">
                                sec
                            </div>
                        </div>
                        <p className="text-xs text-slate-400">
                            Recording duration (5-300 seconds, 7 WebSDR receivers)
                        </p>
                    </div>

                    {/* Info Box */}
                    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-2">
                        <p className="text-sm text-slate-300 font-semibold">Acquisition Details:</p>
                        <ul className="text-xs text-slate-400 space-y-1 pl-4">
                            <li>• <strong>Receivers:</strong> 7 WebSDR (Northwestern Italy)</li>
                            <li>• <strong>Frequency:</strong> {frequency.toFixed(3)} MHz</li>
                            <li>• <strong>Duration:</strong> {duration} seconds ({Math.ceil(duration * 7 / 60)} min for 7 receivers)</li>
                            <li>• <strong>Output:</strong> IQ data + SNR metadata</li>
                        </ul>
                    </div>

                    {/* Submit Button */}
                    <Button
                        type="submit"
                        disabled={isSubmitting || !sessionName.trim()}
                        className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 flex items-center justify-center gap-2"
                    >
                        <Zap className="w-4 h-4" />
                        {isSubmitting ? 'Starting Acquisition...' : 'Start Acquisition'}
                    </Button>

                    {/* Tips */}
                    <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-3">
                        <p className="text-xs text-blue-300">
                            💡 <strong>Tip:</strong> Sessions are queued and processed sequentially. You'll see the status update in real-time.
                        </p>
                    </div>
                </form>
            </CardContent>
        </Card>
    );
};

export default RecordingSessionCreator;
