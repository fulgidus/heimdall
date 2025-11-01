use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub name: String,
    pub driver_version: String,
    pub cuda_version: String,
    pub memory_total_mb: u64,
    pub memory_free_mb: u64,
    pub memory_used_mb: u64,
    pub utilization_percent: u32,
}

/// Check for available GPU and return information
#[tauri::command]
pub async fn check_gpu() -> Result<GpuInfo, String> {
    log::info!("Checking GPU availability");
    
    // Try nvidia-smi first (most common for CUDA)
    let gpu_info = check_nvidia_smi().unwrap_or_else(|_| {
        // If nvidia-smi fails, return default "not available" info
        GpuInfo {
            available: false,
            name: "No GPU detected".to_string(),
            driver_version: "N/A".to_string(),
            cuda_version: "N/A".to_string(),
            memory_total_mb: 0,
            memory_free_mb: 0,
            memory_used_mb: 0,
            utilization_percent: 0,
        }
    });
    
    Ok(gpu_info)
}

/// Get current GPU usage statistics
#[tauri::command]
pub async fn get_gpu_usage() -> Result<GpuInfo, String> {
    log::debug!("Getting GPU usage");
    check_gpu().await
}

// Helper function to check GPU via nvidia-smi
fn check_nvidia_smi() -> Result<GpuInfo, String> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;
    
    if !output.status.success() {
        return Err("nvidia-smi command failed".to_string());
    }
    
    let output_str = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = output_str.trim().split(',').map(|s| s.trim()).collect();
    
    if parts.len() < 6 {
        return Err("Invalid nvidia-smi output format".to_string());
    }
    
    Ok(GpuInfo {
        available: true,
        name: parts[0].to_string(),
        driver_version: parts[1].to_string(),
        cuda_version: "N/A".to_string(), // Would need separate query
        memory_total_mb: parts[2].parse().unwrap_or(0),
        memory_free_mb: parts[3].parse().unwrap_or(0),
        memory_used_mb: parts[4].parse().unwrap_or(0),
        utilization_percent: parts[5].parse().unwrap_or(0),
    })
}
