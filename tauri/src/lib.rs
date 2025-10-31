mod commands;

use commands::{data_collection, training, gpu, settings};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_log::Builder::default()
      .level(log::LevelFilter::Info)
      .build())
    .plugin(tauri_plugin_shell::init())
    .setup(|_app| {
      log::info!("Heimdall SDR Desktop starting...");
      log::info!("Application running in {} mode", if cfg!(debug_assertions) { "debug" } else { "release" });
      
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      // Data collection commands
      data_collection::start_data_collection,
      data_collection::stop_data_collection,
      data_collection::get_collection_status,
      // Training commands
      training::start_training,
      training::stop_training,
      training::get_training_status,
      // GPU commands
      gpu::check_gpu,
      gpu::get_gpu_usage,
      // Settings commands
      settings::load_settings,
      settings::save_settings,
      settings::reset_settings,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
