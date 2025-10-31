mod commands;

use commands::{data_collection, training, gpu, settings, import_export};
use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_log::Builder::default()
      .level(log::LevelFilter::Info)
      .build())
    .plugin(tauri_plugin_shell::init())
    .plugin(tauri_plugin_dialog::init())
    .setup(|app| {
      log::info!("Heimdall SDR Desktop starting...");
      log::info!("Application running in {} mode", if cfg!(debug_assertions) { "debug" } else { "release" });
      
      // Check for command-line arguments (file path from OS file association)
      let args: Vec<String> = std::env::args().collect();
      log::info!("Application started with {} arguments", args.len());
      
      // If launched with a .heimdall file, emit event to frontend
      if args.len() > 1 {
        for arg in args.iter().skip(1) {
          if arg.ends_with(".heimdall") && std::path::Path::new(arg).exists() {
            log::info!("Detected .heimdall file from command line: {}", arg);
            let file_path = arg.clone();
            let app_handle = app.handle().clone();
            
            // Emit event to frontend after a short delay to ensure UI is ready
            std::thread::spawn(move || {
              std::thread::sleep(std::time::Duration::from_millis(1000));
              let _ = app_handle.emit("open-heimdall-file", file_path);
            });
            break;
          }
        }
      }
      
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
      // Import/Export commands
      import_export::save_heimdall_file,
      import_export::load_heimdall_file,
      import_export::load_heimdall_file_from_path,
      import_export::save_heimdall_file_to_path,
      import_export::get_default_export_path,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
