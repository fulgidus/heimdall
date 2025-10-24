export interface RecordingSession {
    id: number;
    session_name: string;
    frequency_mhz: number;
    duration_seconds: number;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
    celery_task_id: string | null;
    result_metadata: Record<string, any> | null;
    minio_path: string | null;
    error_message: string | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    websdrs_enabled: number;
}

export interface SessionStatus {
    session_id: number;
    status: string;
    progress: number;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    error_message: string | null;
}
