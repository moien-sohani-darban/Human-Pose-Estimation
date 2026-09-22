fn main() {
    let manifest_directory = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );
    let packaged_sidecar = manifest_directory
        .join("..")
        .join("python-engine")
        .join("dist")
        .join("hpe-python-sidecar")
        .join("hpe-python-sidecar.exe");
    println!("cargo:rerun-if-changed={}", packaged_sidecar.display());
    if std::env::var("PROFILE").as_deref() == Ok("release") && !packaged_sidecar.is_file() {
        panic!(
            "Task 13 production runtime is missing at {}. Run scripts/build-python-sidecar.ps1 before the Tauri release build.",
            packaged_sidecar.display()
        );
    }
    tauri_build::build()
}
