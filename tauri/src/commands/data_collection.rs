use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DataCollectionConfig {
    pub frequency: f64,
    pub duration_seconds: u32,
    pub websdrs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataCollectionStatus {
    pub is_running: bool,
    pub progress: f32,
    pub current_websdr: Option<String>,
    pub message: String,
}

/// Start RF data collection from configured WebSDRs
/// This is an IPC command that can be called from the frontend
#[tauri::command]
pub async fn start_data_collection(config: DataCollectionConfig) -> Result<String, String> {
    log::info!("Starting data collection with config: {:?}", config);
    
    // In a real implementation, this would communicate with the backend service
    // For now, return success to indicate the command structure works
    Ok(format!(
        "Data collection started for {} WebSDRs at {:.2} MHz for {} seconds",
        config.websdrs.len(),
        config.frequency,
        config.duration_seconds
    ))
}

/// Stop ongoing RF data collection
#[tauri::command]
pub async fn stop_data_collection() -> Result<String, String> {
    log::info!("Stopping data collection");
    Ok("Data collection stopped".to_string())
}

/// Get current data collection status
#[tauri::command]
pub async fn get_collection_status() -> Result<DataCollectionStatus, String> {
    log::debug!("Getting collection status");
    
    Ok(DataCollectionStatus {
        is_running: false,
        progress: 0.0,
        current_websdr: None,
        message: "Ready".to_string(),
    })
}
