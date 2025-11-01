use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub model_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub is_running: bool,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub loss: f64,
    pub accuracy: f64,
    pub message: String,
}

/// Start ML model training
#[tauri::command]
pub async fn start_training(config: TrainingConfig) -> Result<String, String> {
    log::info!("Starting training with config: {:?}", config);
    
    Ok(format!(
        "Training started: {} epochs, batch size {}, learning rate {:.6}",
        config.epochs,
        config.batch_size,
        config.learning_rate
    ))
}

/// Stop ongoing training
#[tauri::command]
pub async fn stop_training() -> Result<String, String> {
    log::info!("Stopping training");
    Ok("Training stopped".to_string())
}

/// Get current training status
#[tauri::command]
pub async fn get_training_status() -> Result<TrainingStatus, String> {
    log::debug!("Getting training status");
    
    Ok(TrainingStatus {
        is_running: false,
        current_epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        accuracy: 0.0,
        message: "Ready".to_string(),
    })
}
