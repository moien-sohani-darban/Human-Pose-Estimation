mod sidecar;

use sidecar::{
    BackendsResult, BridgeError, EstimatePoseRequest, PingResult, PoseResultDto,
    PythonSidecarManager,
};
use tauri::{Manager, RunEvent, State};

async fn run_blocking<T, F>(operation: F) -> Result<T, BridgeError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BridgeError> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(operation)
        .await
        .map_err(|error| BridgeError {
            kind: "sidecar_task_failed".to_owned(),
            code: None,
            message: error.to_string(),
        })?
}

#[tauri::command]
async fn python_sidecar_ping(
    manager: State<'_, PythonSidecarManager>,
) -> Result<PingResult, BridgeError> {
    let manager = manager.inner().clone();
    run_blocking(move || manager.ping()).await
}

#[tauri::command]
async fn python_get_backends(
    manager: State<'_, PythonSidecarManager>,
) -> Result<BackendsResult, BridgeError> {
    let manager = manager.inner().clone();
    run_blocking(move || manager.get_backends()).await
}

#[tauri::command]
async fn python_estimate_pose(
    manager: State<'_, PythonSidecarManager>,
    request: EstimatePoseRequest,
) -> Result<PoseResultDto, BridgeError> {
    let manager = manager.inner().clone();
    run_blocking(move || manager.estimate(request)).await
}

#[tauri::command]
async fn python_shutdown_sidecar(
    manager: State<'_, PythonSidecarManager>,
) -> Result<(), BridgeError> {
    let manager = manager.inner().clone();
    run_blocking(move || manager.shutdown()).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(PythonSidecarManager::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            python_sidecar_ping,
            python_get_backends,
            python_estimate_pose,
            python_shutdown_sidecar
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    application.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            let manager = app_handle.state::<PythonSidecarManager>();
            if let Err(error) = manager.shutdown() {
                eprintln!("Failed to stop Python sidecar during app exit: {error}");
            }
        }
    });
}
