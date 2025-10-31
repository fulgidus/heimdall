use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppSettings {
    pub api_url: String,
    pub websocket_url: String,
    pub mapbox_token: String,
    pub auto_start_backend: bool,
    pub backend_port: u16,
    pub enable_gpu: bool,
    pub theme: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:8000".to_string(),
            websocket_url: "ws://localhost:80/ws".to_string(),
            mapbox_token: String::new(),
            auto_start_backend: false,
            backend_port: 8000,
            enable_gpu: true,
            theme: "light".to_string(),
        }
    }
}

/// Get settings file path (platform-specific)
fn get_settings_path() -> Result<PathBuf, String> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| "Could not find config directory".to_string())?;
    
    let app_config_dir = config_dir.join("heimdall-sdr");
    
    // Create directory if it doesn't exist
    if !app_config_dir.exists() {
        fs::create_dir_all(&app_config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }
    
    Ok(app_config_dir.join("settings.json"))
}

/// Load application settings from file
#[tauri::command]
pub async fn load_settings() -> Result<AppSettings, String> {
    log::info!("Loading application settings");
    
    let settings_path = get_settings_path()?;
    
    if !settings_path.exists() {
        log::info!("Settings file not found, using defaults");
        return Ok(AppSettings::default());
    }
    
    let contents = fs::read_to_string(&settings_path)
        .map_err(|e| format!("Failed to read settings file: {}", e))?;
    
    let settings: AppSettings = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse settings: {}", e))?;
    
    log::info!("Settings loaded successfully");
    Ok(settings)
}

/// Save application settings to file
#[tauri::command]
pub async fn save_settings(settings: AppSettings) -> Result<String, String> {
    log::info!("Saving application settings");
    
    let settings_path = get_settings_path()?;
    
    let json = serde_json::to_string_pretty(&settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    fs::write(&settings_path, json)
        .map_err(|e| format!("Failed to write settings file: {}", e))?;
    
    log::info!("Settings saved successfully to {:?}", settings_path);
    Ok(format!("Settings saved to {:?}", settings_path))
}

/// Reset settings to defaults
#[tauri::command]
pub async fn reset_settings() -> Result<AppSettings, String> {
    log::info!("Resetting settings to defaults");
    
    let settings = AppSettings::default();
    save_settings(settings.clone()).await?;
    
    Ok(settings)
}
