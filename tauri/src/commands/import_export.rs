use serde::{Deserialize, Serialize};
use std::fs;
use tauri_plugin_dialog::DialogExt;

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveFileRequest {
    pub content: String,
    pub filename: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveFileResponse {
    pub success: bool,
    pub message: String,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadFileRequest {
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadFileResponse {
    pub success: bool,
    pub content: Option<String>,
    pub message: String,
}

/// Save .heimdall file to local filesystem with file dialog
#[tauri::command]
pub async fn save_heimdall_file(
    app: tauri::AppHandle,
    content: String,
    default_filename: Option<String>,
) -> Result<SaveFileResponse, String> {
    log::info!("save_heimdall_file command called");
    
    let filename = default_filename.unwrap_or_else(|| {
        format!("heimdall-export-{}.heimdall", 
                chrono::Local::now().format("%Y%m%d-%H%M%S"))
    });
    
    // Use native file dialog
    let file_path = tauri_plugin_dialog::FileDialogBuilder::new(app.dialog().clone())
        .set_title("Save Heimdall Export File")
        .add_filter("Heimdall Files", &["heimdall"])
        .add_filter("All Files", &["*"])
        .set_file_name(&filename)
        .blocking_save_file();
    
    match file_path {
        Some(path) => {
            if let Some(path_buf) = path.as_path() {
                match fs::write(path_buf, content) {
                    Ok(_) => {
                        log::info!("File saved successfully to: {:?}", path_buf);
                        Ok(SaveFileResponse {
                            success: true,
                            message: "File saved successfully".to_string(),
                            path: Some(path_buf.to_string_lossy().to_string()),
                        })
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to write file: {}", e);
                        log::error!("{}", error_msg);
                        Ok(SaveFileResponse {
                            success: false,
                            message: error_msg,
                            path: None,
                        })
                    }
                }
            } else {
                Ok(SaveFileResponse {
                    success: false,
                    message: "Invalid file path".to_string(),
                    path: None,
                })
            }
        }
        None => {
            log::info!("User cancelled file save dialog");
            Ok(SaveFileResponse {
                success: false,
                message: "File save cancelled by user".to_string(),
                path: None,
            })
        }
    }
}

/// Load .heimdall file from local filesystem with file dialog
#[tauri::command]
pub async fn load_heimdall_file(app: tauri::AppHandle) -> Result<LoadFileResponse, String> {
    log::info!("load_heimdall_file command called");
    
    // Use native file dialog
    let file_path = tauri_plugin_dialog::FileDialogBuilder::new(app.dialog().clone())
        .set_title("Open Heimdall Export File")
        .add_filter("Heimdall Files", &["heimdall"])
        .add_filter("All Files", &["*"])
        .blocking_pick_file();
    
    match file_path {
        Some(path) => {
            if let Some(path_buf) = path.as_path() {
                match fs::read_to_string(path_buf) {
                    Ok(content) => {
                        log::info!("File loaded successfully from: {:?}", path_buf);
                        Ok(LoadFileResponse {
                            success: true,
                            content: Some(content),
                            message: format!("File loaded: {}", path_buf.to_string_lossy()),
                        })
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to read file: {}", e);
                        log::error!("{}", error_msg);
                        Ok(LoadFileResponse {
                            success: false,
                            content: None,
                            message: error_msg,
                        })
                    }
                }
            } else {
                Ok(LoadFileResponse {
                    success: false,
                    content: None,
                    message: "Invalid file path".to_string(),
                })
            }
        }
        None => {
            log::info!("User cancelled file open dialog");
            Ok(LoadFileResponse {
                success: false,
                content: None,
                message: "File open cancelled by user".to_string(),
            })
        }
    }
}

/// Load .heimdall file from specific path (without dialog)
#[tauri::command]
pub async fn load_heimdall_file_from_path(
    _app: tauri::AppHandle,
    path: String,
) -> Result<LoadFileResponse, String> {
    log::info!("load_heimdall_file_from_path command called with path: {}", path);
    
    match fs::read_to_string(&path) {
        Ok(content) => {
            log::info!("File loaded successfully from: {}", path);
            Ok(LoadFileResponse {
                success: true,
                content: Some(content),
                message: format!("File loaded: {}", path),
            })
        }
        Err(e) => {
            let error_msg = format!("Failed to read file: {}", e);
            log::error!("{}", error_msg);
            Ok(LoadFileResponse {
                success: false,
                content: None,
                message: error_msg,
            })
        }
    }
}

/// Save .heimdall file to specific path (without dialog)
#[tauri::command]
pub async fn save_heimdall_file_to_path(
    _app: tauri::AppHandle,
    content: String,
    path: String,
) -> Result<SaveFileResponse, String> {
    log::info!("save_heimdall_file_to_path command called with path: {}", path);
    
    match fs::write(&path, content) {
        Ok(_) => {
            log::info!("File saved successfully to: {}", path);
            Ok(SaveFileResponse {
                success: true,
                message: "File saved successfully".to_string(),
                path: Some(path),
            })
        }
        Err(e) => {
            let error_msg = format!("Failed to write file: {}", e);
            log::error!("{}", error_msg);
            Ok(SaveFileResponse {
                success: false,
                message: error_msg,
                path: None,
            })
        }
    }
}

/// Get default documents directory for saving exports
#[tauri::command]
pub async fn get_default_export_path() -> Result<String, String> {
    match dirs::document_dir() {
        Some(path) => Ok(path.to_string_lossy().to_string()),
        None => Ok(std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string())),
    }
}
