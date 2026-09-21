use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    env,
    ffi::OsString,
    fmt,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc, Mutex, MutexGuard},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const PROTOCOL_VERSION: u8 = 1;
const DIAGNOSTIC_LIMIT: usize = 32;
pub const SHORT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
pub const ESTIMATE_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BridgeError {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
}

impl BridgeError {
    fn new(kind: &str, message: impl Into<String>) -> Self {
        Self {
            kind: kind.to_owned(),
            code: None,
            message: message.into(),
        }
    }

    fn python(code: String, message: String) -> Self {
        Self {
            kind: "python_protocol_error".to_owned(),
            code: Some(code),
            message,
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for BridgeError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PingResult {
    pub status: String,
    pub protocol_version: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendsResult {
    pub available: Vec<String>,
    pub unavailable: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<BTreeMap<String, ModelAssetStatusDto>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelAssetStatusDto {
    pub backend: String,
    pub model_name: String,
    pub default_path: String,
    pub display_path: String,
    pub exists: bool,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoundingBoxDto {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeypointDto {
    pub index: u32,
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersonPoseDto {
    pub person_id: u32,
    pub keypoints: Vec<KeypointDto>,
    pub bbox: Option<BoundingBoxDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoseResultDto {
    pub success: bool,
    pub backend: String,
    pub people: Vec<PersonPoseDto>,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub processing_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EstimatePoseRequest {
    pub backend: String,
    pub image_path: PathBuf,
    pub model_path: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ProtocolRequest {
    id: u64,
    protocol_version: u8,
    #[serde(flatten)]
    command: ProtocolCommand,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ProtocolCommand {
    Ping,
    GetBackends,
    Estimate {
        backend: String,
        image_path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        backend_config: Option<BackendConfig>,
    },
    Shutdown,
}

#[derive(Debug, Serialize)]
struct BackendConfig {
    model_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ProtocolResponse<T> {
    id: Value,
    ok: bool,
    result: Option<T>,
    error: Option<PythonError>,
}

#[derive(Debug, Deserialize)]
struct PythonError {
    code: String,
    message: String,
}

#[derive(Debug, Clone)]
struct LaunchConfig {
    executable: PathBuf,
    working_directory: PathBuf,
    arguments: Vec<OsString>,
}

impl LaunchConfig {
    fn discover_development() -> Self {
        Self::from_sources(
            env::var_os("HPE_PYTHON_EXECUTABLE"),
            env::var_os("HPE_PYTHON_ENGINE_DIR"),
        )
    }

    fn from_sources(executable: Option<OsString>, engine_dir: Option<OsString>) -> Self {
        let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
        let default_engine_dir = crate_directory.join("..").join("python-engine");
        let working_directory = engine_dir
            .map(PathBuf::from)
            .map(|path| resolve_from(crate_directory, path))
            .unwrap_or(default_engine_dir);
        let executable = executable
            .map(PathBuf::from)
            .map(|path| resolve_from(crate_directory, path))
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    working_directory
                        .join(".venv")
                        .join("Scripts")
                        .join("python.exe")
                } else {
                    working_directory.join(".venv").join("bin").join("python")
                }
            });
        Self {
            executable,
            working_directory,
            arguments: vec![OsString::from("-m"), OsString::from("app.main")],
        }
    }
}

fn resolve_from(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

enum StdoutEvent {
    Line(String),
    Eof,
    ReadFailed(String),
}

struct SidecarProcess {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: mpsc::Receiver<StdoutEvent>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
}

struct ManagerInner {
    child: Option<SidecarProcess>,
    next_request_id: u64,
    spawn_count: u64,
}

#[derive(Clone)]
pub struct PythonSidecarManager {
    inner: Arc<Mutex<ManagerInner>>,
    launch: LaunchConfig,
    diagnostics: Arc<Mutex<VecDeque<String>>>,
    short_timeout: Duration,
    estimate_timeout: Duration,
}

impl Default for PythonSidecarManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonSidecarManager {
    pub fn new() -> Self {
        Self::with_config(
            LaunchConfig::discover_development(),
            SHORT_REQUEST_TIMEOUT,
            ESTIMATE_REQUEST_TIMEOUT,
        )
    }

    fn with_config(
        launch: LaunchConfig,
        short_timeout: Duration,
        estimate_timeout: Duration,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ManagerInner {
                child: None,
                next_request_id: 1,
                spawn_count: 0,
            })),
            launch,
            diagnostics: Arc::new(Mutex::new(VecDeque::new())),
            short_timeout,
            estimate_timeout,
        }
    }

    pub fn ping(&self) -> Result<PingResult, BridgeError> {
        self.request(ProtocolCommand::Ping, self.short_timeout)
    }

    #[cfg(test)]
    fn start(&self) -> Result<(), BridgeError> {
        let mut inner = self.lock_inner()?;
        self.ensure_process_started(&mut inner)
    }

    pub fn get_backends(&self) -> Result<BackendsResult, BridgeError> {
        self.request(ProtocolCommand::GetBackends, self.short_timeout)
    }

    pub fn estimate(&self, request: EstimatePoseRequest) -> Result<PoseResultDto, BridgeError> {
        let backend_config = request
            .model_path
            .map(|model_path| BackendConfig { model_path });
        self.request(
            ProtocolCommand::Estimate {
                backend: request.backend,
                image_path: request.image_path,
                backend_config,
            },
            self.estimate_timeout,
        )
    }

    pub fn shutdown(&self) -> Result<(), BridgeError> {
        let mut inner = self.lock_inner()?;
        if !Self::child_is_alive(&mut inner)? {
            Self::clear_process(&mut inner);
            return Ok(());
        }

        let request_id = match Self::next_id(&mut inner) {
            Ok(id) => id,
            Err(error) => {
                Self::terminate_process(&mut inner);
                return Err(error);
            }
        };
        let request = ProtocolRequest {
            id: request_id,
            protocol_version: PROTOCOL_VERSION,
            command: ProtocolCommand::Shutdown,
        };
        if let Err(error) = self.exchange::<Value>(&mut inner, request, self.short_timeout) {
            Self::terminate_process(&mut inner);
            return Err(error);
        }

        let deadline = Instant::now() + self.short_timeout;
        while Instant::now() < deadline {
            let exited = match inner.child.as_mut() {
                Some(process) => process
                    .child
                    .try_wait()
                    .map_err(|error| BridgeError::new("sidecar_read_failed", error.to_string()))?,
                None => return Ok(()),
            };
            if exited.is_some() {
                Self::clear_process(&mut inner);
                return Ok(());
            }
            thread::sleep(Duration::from_millis(10));
        }
        Self::terminate_process(&mut inner);
        Err(BridgeError::new(
            "sidecar_timeout",
            "Python sidecar did not exit after shutdown",
        ))
    }

    #[cfg(test)]
    fn recent_diagnostics(&self) -> Vec<String> {
        self.diagnostics
            .lock()
            .map(|items| items.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn request<T: DeserializeOwned>(
        &self,
        command: ProtocolCommand,
        timeout: Duration,
    ) -> Result<T, BridgeError> {
        let mut inner = self.lock_inner()?;
        self.ensure_process_started(&mut inner)?;
        let request_id = match Self::next_id(&mut inner) {
            Ok(id) => id,
            Err(error) => {
                Self::terminate_process(&mut inner);
                return Err(error);
            }
        };
        let request = ProtocolRequest {
            id: request_id,
            protocol_version: PROTOCOL_VERSION,
            command,
        };
        self.exchange(&mut inner, request, timeout)
    }

    fn exchange<T: DeserializeOwned>(
        &self,
        inner: &mut ManagerInner,
        request: ProtocolRequest,
        timeout: Duration,
    ) -> Result<T, BridgeError> {
        let request_id = request.id;
        let mut bytes = serde_json::to_vec(&request).map_err(|error| {
            BridgeError::new("protocol_serialization_failed", error.to_string())
        })?;
        bytes.push(b'\n');

        let process = inner.child.as_mut().ok_or_else(|| {
            BridgeError::new("sidecar_not_running", "Python sidecar is not running")
        })?;
        if let Err(error) = process
            .stdin
            .write_all(&bytes)
            .and_then(|_| process.stdin.flush())
        {
            let bridge_error = BridgeError::new("sidecar_write_failed", error.to_string());
            Self::terminate_process(inner);
            return Err(bridge_error);
        }

        let event = process.stdout_rx.recv_timeout(timeout);
        let line = match event {
            Ok(StdoutEvent::Line(line)) => line,
            Ok(StdoutEvent::Eof) => {
                Self::terminate_process(inner);
                return Err(BridgeError::new(
                    "sidecar_exited",
                    "Python sidecar exited before responding",
                ));
            }
            Ok(StdoutEvent::ReadFailed(message)) => {
                Self::terminate_process(inner);
                return Err(BridgeError::new("sidecar_read_failed", message));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                Self::terminate_process(inner);
                return Err(BridgeError::new(
                    "sidecar_timeout",
                    format!("Python sidecar request {request_id} timed out"),
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Self::terminate_process(inner);
                return Err(BridgeError::new(
                    "sidecar_exited",
                    "Python sidecar response channel closed",
                ));
            }
        };

        match parse_response(&line, request_id) {
            Ok(value) => Ok(value),
            Err(error) => {
                if error.kind != "python_protocol_error" {
                    Self::terminate_process(inner);
                }
                Err(error)
            }
        }
    }

    fn ensure_process_started(&self, inner: &mut ManagerInner) -> Result<(), BridgeError> {
        if Self::child_is_alive(inner)? {
            return Ok(());
        }
        Self::clear_process(inner);
        let mut command = Command::new(&self.launch.executable);
        command
            .args(&self.launch.arguments)
            .current_dir(&self.launch.working_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().map_err(|error| {
            BridgeError::new(
                "sidecar_spawn_failed",
                format!(
                    "Could not launch Python at {} from {}: {error}",
                    self.launch.executable.display(),
                    self.launch.working_directory.display()
                ),
            )
        })?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(BridgeError::new(
                "sidecar_spawn_failed",
                "Python process streams were not all piped",
            ));
        };

        let (stdout_tx, stdout_rx) = mpsc::channel();
        let stdout_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = stdout_tx.send(StdoutEvent::Eof);
                        break;
                    }
                    Ok(_) => {
                        while line.ends_with(['\r', '\n']) {
                            line.pop();
                        }
                        if stdout_tx.send(StdoutEvent::Line(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = stdout_tx.send(StdoutEvent::ReadFailed(error.to_string()));
                        break;
                    }
                }
            }
        });
        let diagnostics = Arc::clone(&self.diagnostics);
        let stderr_thread = thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) => {
                        eprintln!("[python-sidecar] {line}");
                        if let Ok(mut recent) = diagnostics.lock() {
                            if recent.len() == DIAGNOSTIC_LIMIT {
                                recent.pop_front();
                            }
                            recent.push_back(line);
                        }
                    }
                    Err(error) => {
                        eprintln!("[python-sidecar] stderr read failed: {error}");
                        break;
                    }
                }
            }
        });
        inner.child = Some(SidecarProcess {
            child,
            stdin,
            stdout_rx,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
        });
        inner.spawn_count += 1;
        Ok(())
    }

    fn child_is_alive(inner: &mut ManagerInner) -> Result<bool, BridgeError> {
        match inner.child.as_mut() {
            None => Ok(false),
            Some(process) => process
                .child
                .try_wait()
                .map(|status| status.is_none())
                .map_err(|error| BridgeError::new("sidecar_read_failed", error.to_string())),
        }
    }

    fn next_id(inner: &mut ManagerInner) -> Result<u64, BridgeError> {
        let id = inner.next_request_id;
        inner.next_request_id = inner.next_request_id.checked_add(1).ok_or_else(|| {
            BridgeError::new("sidecar_state_failed", "Request ID space was exhausted")
        })?;
        Ok(id)
    }

    fn terminate_process(inner: &mut ManagerInner) {
        if let Some(mut process) = inner.child.take() {
            if process.child.try_wait().ok().flatten().is_none() {
                let _ = process.child.kill();
            }
            let _ = process.child.wait();
            drop(process.stdin);
            if let Some(thread) = process.stdout_thread.take() {
                let _ = thread.join();
            }
            if let Some(thread) = process.stderr_thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn clear_process(inner: &mut ManagerInner) {
        if let Some(mut process) = inner.child.take() {
            let _ = process.child.wait();
            drop(process.stdin);
            if let Some(thread) = process.stdout_thread.take() {
                let _ = thread.join();
            }
            if let Some(thread) = process.stderr_thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, ManagerInner>, BridgeError> {
        self.inner.lock().map_err(|_| {
            BridgeError::new("sidecar_state_failed", "Sidecar manager lock is poisoned")
        })
    }

    #[cfg(test)]
    fn spawn_count(&self) -> u64 {
        self.inner
            .lock()
            .map(|inner| inner.spawn_count)
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn is_running(&self) -> bool {
        self.inner
            .lock()
            .map(|mut inner| Self::child_is_alive(&mut inner).unwrap_or(false))
            .unwrap_or(false)
    }
}

impl Drop for PythonSidecarManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            let _ = self.shutdown();
        }
    }
}

fn parse_response<T: DeserializeOwned>(line: &str, expected_id: u64) -> Result<T, BridgeError> {
    if line.is_empty() {
        return Err(BridgeError::new(
            "protocol_parse_failed",
            "Python sidecar returned an empty response line",
        ));
    }
    let response: ProtocolResponse<Value> = serde_json::from_str(line).map_err(|error| {
        BridgeError::new(
            "protocol_parse_failed",
            format!("Invalid response JSON: {error}"),
        )
    })?;
    if response.id.as_u64() != Some(expected_id) {
        return Err(BridgeError::new(
            "protocol_id_mismatch",
            format!(
                "Expected response id {expected_id}, received {}",
                response.id
            ),
        ));
    }
    match (response.ok, response.result, response.error) {
        (true, Some(result), None) => serde_json::from_value(result).map_err(|error| {
            BridgeError::new(
                "protocol_parse_failed",
                format!("Invalid response result: {error}"),
            )
        }),
        (false, None, Some(error)) => Err(BridgeError::python(error.code, error.message)),
        _ => Err(BridgeError::new(
            "protocol_parse_failed",
            "Response envelope has inconsistent ok/result/error fields",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

    fn helper_config(script: &str, extra_args: &[&Path]) -> LaunchConfig {
        let mut config = LaunchConfig::discover_development();
        config.arguments = vec![
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ];
        config
            .arguments
            .extend(extra_args.iter().map(|path| path.as_os_str().to_owned()));
        config
    }

    fn temp_marker(name: &str) -> PathBuf {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!("hpe-task08-{}-{name}-{id}", std::process::id()))
    }

    #[test]
    fn launch_configuration_honors_explicit_sources() {
        let config = LaunchConfig::from_sources(
            Some(OsString::from(r"C:\Python\python.exe")),
            Some(OsString::from(r"C:\Project\python-engine")),
        );
        assert_eq!(config.executable, PathBuf::from(r"C:\Python\python.exe"));
        assert_eq!(
            config.working_directory,
            PathBuf::from(r"C:\Project\python-engine")
        );
        assert_eq!(config.arguments, ["-m", "app.main"]);

        let relative = LaunchConfig::from_sources(
            Some(OsString::from(r"tools\python.exe")),
            Some(OsString::from(r"..\python-engine")),
        );
        let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(
            relative.executable,
            crate_directory.join(r"tools\python.exe")
        );
        assert_eq!(
            relative.working_directory,
            crate_directory.join(r"..\python-engine")
        );
    }

    #[test]
    fn development_launch_uses_project_venv_and_module_entrypoint() {
        let config = LaunchConfig::from_sources(None, None);
        let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
        let engine_directory = crate_directory.join("..").join("python-engine");
        assert_eq!(
            config.executable,
            engine_directory
                .join(".venv")
                .join("Scripts")
                .join("python.exe")
        );
        assert_eq!(config.working_directory, engine_directory);
        assert_eq!(config.arguments, ["-m", "app.main"]);
    }

    fn echo_script() -> &'static str {
        r#"import json,sys
for line in sys.stdin:
 r=json.loads(line); t=r['type']; rid=r['id']
 if t=='ping': result={'status':'ready','protocol_version':1}
 elif t=='get_backends': result={'available':['mediapipe','yolo'],'unavailable':['mmpose']}
 elif t=='estimate': result={'success':True,'backend':r['backend'],'people':[],'image_width':None,'image_height':None,'processing_time_ms':None}
 elif t=='shutdown': result={'status':'shutdown'}
 print(json.dumps({'id':rid,'ok':True,'result':result}),flush=True)
 if t=='shutdown': break"#
    }

    #[test]
    fn serializes_all_protocol_commands_with_version_and_ids() {
        let requests = [
            ProtocolRequest {
                id: 1,
                protocol_version: 1,
                command: ProtocolCommand::Ping,
            },
            ProtocolRequest {
                id: 2,
                protocol_version: 1,
                command: ProtocolCommand::GetBackends,
            },
            ProtocolRequest {
                id: 3,
                protocol_version: 1,
                command: ProtocolCommand::Estimate {
                    backend: "yolo".into(),
                    image_path: PathBuf::from(r"C:\images\person.jpg"),
                    backend_config: Some(BackendConfig {
                        model_path: PathBuf::from(r"C:\models\pose.pt"),
                    }),
                },
            },
            ProtocolRequest {
                id: 4,
                protocol_version: 1,
                command: ProtocolCommand::Shutdown,
            },
        ];
        let values: Vec<Value> = requests
            .iter()
            .map(|request| serde_json::to_value(request).unwrap())
            .collect();
        assert_eq!(
            values[0],
            serde_json::json!({"id":1,"protocol_version":1,"type":"ping"})
        );
        assert_eq!(values[1]["type"], "get_backends");
        assert_eq!(values[2]["image_path"], r"C:\images\person.jpg");
        assert_eq!(
            values[2]["backend_config"]["model_path"],
            r"C:\models\pose.pt"
        );
        assert_eq!(values[3]["type"], "shutdown");
    }

    #[test]
    fn parses_success_error_malformed_and_mismatched_responses() {
        let ping: PingResult = parse_response(
            r#"{"id":1,"ok":true,"result":{"status":"ready","protocol_version":1}}"#,
            1,
        )
        .unwrap();
        assert_eq!(ping.status, "ready");
        let error = parse_response::<Value>(
            r#"{"id":2,"ok":false,"error":{"code":"model_asset_not_found","message":"missing"}}"#,
            2,
        )
        .unwrap_err();
        assert_eq!(error.kind, "python_protocol_error");
        assert_eq!(error.code.as_deref(), Some("model_asset_not_found"));
        assert_eq!(
            parse_response::<Value>("not-json", 1).unwrap_err().kind,
            "protocol_parse_failed"
        );
        assert_eq!(
            parse_response::<Value>(r#"{"id":9,"ok":true,"result":{}}"#, 1)
                .unwrap_err()
                .kind,
            "protocol_id_mismatch"
        );
    }

    #[test]
    fn deserializes_pose_dto_with_nulls_and_multiple_people() {
        let json = r#"{"id":1,"ok":true,"result":{"success":true,"backend":"yolo","people":[{"person_id":0,"keypoints":[{"index":0,"name":"nose","x":0.1,"y":0.2,"z":null,"confidence":0.9}],"bbox":{"x":0.0,"y":0.1,"width":0.5,"height":0.8}},{"person_id":1,"keypoints":[{"index":0,"name":"nose","x":0.3,"y":0.4,"z":null,"confidence":null}],"bbox":null}],"image_width":640,"image_height":480,"processing_time_ms":12.5}}"#;
        let result: PoseResultDto = parse_response(json, 1).unwrap();
        assert_eq!(result.people.len(), 2);
        assert_eq!(result.people[0].keypoints[0].z, None);
        assert_eq!(result.people[1].keypoints[0].confidence, None);
        assert_eq!(result.people[1].bbox, None);
    }

    #[test]
    fn backend_dto_parses_additive_model_metadata_and_legacy_shape() {
        let json = r#"{"id":1,"ok":true,"result":{"available":["mediapipe","yolo"],"unavailable":["mmpose"],"models":{"mediapipe":{"backend":"mediapipe","model_name":"mediapipe-pose-landmarker","default_path":"models/mediapipe/pose_landmarker.task","display_path":"models/mediapipe/pose_landmarker.task","exists":true,"size_bytes":42},"yolo":{"backend":"yolo","model_name":"yolo11n-pose","default_path":"models/yolo/yolo11n-pose.pt","display_path":"models/yolo/yolo11n-pose.pt","exists":false,"size_bytes":null}}}}"#;

        let result: BackendsResult = parse_response(json, 1).unwrap();
        let models = result.models.unwrap();

        assert!(models["mediapipe"].exists);
        assert_eq!(models["mediapipe"].size_bytes, Some(42));
        assert_eq!(models["yolo"].display_path, "models/yolo/yolo11n-pose.pt");
        assert_eq!(models["yolo"].size_bytes, None);

        let legacy = r#"{"id":2,"ok":true,"result":{"available":["mediapipe"],"unavailable":[]}}"#;

        let result: BackendsResult = parse_response(legacy, 2).unwrap();
        assert_eq!(result.models, None);
    }

    #[test]
    fn preserves_distinct_python_error_codes() {
        for code in [
            "backend_unavailable",
            "model_asset_not_found",
            "backend_initialization_failed",
            "backend_inference_failed",
        ] {
            let line =
                format!(r#"{{"id":1,"ok":false,"error":{{"code":"{code}","message":"message"}}}}"#);
            let error = parse_response::<Value>(&line, 1).unwrap_err();
            assert_eq!(error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn manager_is_lazy_and_repeated_requests_reuse_one_process() {
        let manager = PythonSidecarManager::with_config(
            helper_config(echo_script(), &[]),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert!(!manager.is_running());
        assert_eq!(manager.spawn_count(), 0);
        assert_eq!(manager.ping().unwrap().status, "ready");
        assert_eq!(
            manager.get_backends().unwrap().available,
            ["mediapipe", "yolo"]
        );
        assert_eq!(manager.spawn_count(), 1);
        manager.shutdown().unwrap();
        assert!(!manager.is_running());
    }

    #[test]
    fn repeated_start_is_idempotent() {
        let manager = PythonSidecarManager::with_config(
            helper_config(echo_script(), &[]),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        manager.start().unwrap();
        manager.start().unwrap();
        manager.start().unwrap();
        assert_eq!(manager.spawn_count(), 1);
        manager.shutdown().unwrap();
    }

    #[test]
    fn estimate_bridge_sends_path_and_deserializes_pose_result() {
        let manager = PythonSidecarManager::with_config(
            helper_config(echo_script(), &[]),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        let result = manager
            .estimate(EstimatePoseRequest {
                backend: "yolo".into(),
                image_path: PathBuf::from(r"C:\images\person.jpg"),
                model_path: Some(PathBuf::from(r"C:\models\pose.pt")),
            })
            .unwrap();
        assert!(result.success);
        assert_eq!(result.backend, "yolo");
        assert!(result.people.is_empty());
        manager.shutdown().unwrap();
    }

    #[test]
    fn spawn_failure_is_typed() {
        let mut config = helper_config(echo_script(), &[]);
        config.executable = PathBuf::from("definitely-missing-python-executable");
        let manager = PythonSidecarManager::with_config(
            config,
            Duration::from_secs(1),
            Duration::from_secs(1),
        );
        assert_eq!(manager.ping().unwrap_err().kind, "sidecar_spawn_failed");
        assert!(!manager.is_running());
    }

    #[test]
    fn malformed_response_and_id_mismatch_are_typed() {
        for (script, expected) in [
            ("import sys; sys.stdin.readline(); print('bad',flush=True)", "protocol_parse_failed"),
            ("import sys; sys.stdin.readline(); print('{\"id\":99,\"ok\":true,\"result\":{}}',flush=True)", "protocol_id_mismatch"),
        ] {
            let manager = PythonSidecarManager::with_config(helper_config(script, &[]), Duration::from_secs(2), Duration::from_secs(2));
            assert_eq!(manager.ping().unwrap_err().kind, expected);
            assert!(!manager.is_running());
        }
    }

    #[test]
    fn unexpected_exit_is_typed_and_next_request_restarts_without_retry() {
        let marker = temp_marker("exit");
        let script = r#"import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); line=sys.stdin.readline()
if not p.exists(): p.write_text('1'); sys.exit(7)
r=json.loads(line); print(json.dumps({'id':r['id'],'ok':True,'result':{'status':'ready','protocol_version':1}}),flush=True)
for line in sys.stdin:
 r=json.loads(line); print(json.dumps({'id':r['id'],'ok':True,'result':{'status':'shutdown'}}),flush=True); break"#;
        let manager = PythonSidecarManager::with_config(
            helper_config(script, &[&marker]),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(manager.ping().unwrap_err().kind, "sidecar_exited");
        assert_eq!(manager.spawn_count(), 1);
        assert_eq!(manager.ping().unwrap().status, "ready");
        assert_eq!(manager.spawn_count(), 2);
        manager.shutdown().unwrap();
        let _ = fs::remove_file(marker);
    }

    #[test]
    fn timeout_kills_desynchronized_child_and_next_request_restarts() {
        let marker = temp_marker("timeout");
        let script = r#"import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); first=not p.exists()
for line in sys.stdin:
 r=json.loads(line)
 if first: p.write_text('1'); time.sleep(2); first=False
 result={'status':'ready','protocol_version':1} if r['type']=='ping' else {'status':'shutdown'}
 print(json.dumps({'id':r['id'],'ok':True,'result':result}),flush=True)
 if r['type']=='shutdown': break"#;
        let manager = PythonSidecarManager::with_config(
            helper_config(script, &[&marker]),
            Duration::from_millis(500),
            Duration::from_millis(500),
        );
        assert_eq!(manager.ping().unwrap_err().kind, "sidecar_timeout");
        assert!(!manager.is_running());
        assert_eq!(manager.ping().unwrap().status, "ready");
        assert_eq!(manager.spawn_count(), 2);
        manager.shutdown().unwrap();
        let _ = fs::remove_file(marker);
    }

    #[test]
    fn stderr_is_drained_separately_and_bounded() {
        let script = format!("import json,sys\nfor line in sys.stdin:\n r=json.loads(line)\n print('diagnostic',file=sys.stderr,flush=True)\n print(json.dumps({{'id':r['id'],'ok':True,'result':{{'status':'ready','protocol_version':1}}}}),flush=True)\n");
        let manager = PythonSidecarManager::with_config(
            helper_config(&script, &[]),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(manager.ping().unwrap().status, "ready");
        for _ in 0..20 {
            if !manager.recent_diagnostics().is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(manager
            .recent_diagnostics()
            .iter()
            .any(|line| line == "diagnostic"));
        assert!(manager.recent_diagnostics().len() <= DIAGNOSTIC_LIMIT);
        PythonSidecarManager::terminate_for_test(&manager);
    }

    #[test]
    fn real_python_bridge_ping_backends_and_shutdown() {
        let manager = PythonSidecarManager::new();
        assert_eq!(
            manager.ping().unwrap(),
            PingResult {
                status: "ready".into(),
                protocol_version: 1
            }
        );
        let backends = manager.get_backends().unwrap();
        assert_eq!(backends.available, ["mediapipe", "yolo"]);
        assert_eq!(backends.unavailable, ["mmpose"]);

        let models = backends.models.expect("real Python metadata is required");

        assert_eq!(
            models.keys().cloned().collect::<Vec<_>>(),
            ["mediapipe", "yolo"]
        );
        assert_eq!(
            models["mediapipe"].display_path,
            "models/mediapipe/pose_landmarker.task"
        );
        assert_eq!(models["yolo"].display_path, "models/yolo/yolo11n-pose.pt");

        let engine_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python-engine");

        for model in models.values() {
            let expected_path = engine_root.join(&model.default_path);

            assert_eq!(model.exists, expected_path.is_file());
            assert_eq!(
                model.size_bytes,
                expected_path.metadata().ok().map(|metadata| metadata.len())
            );
        }

        manager.shutdown().unwrap();
        assert!(!manager.is_running());
    }

    #[test]
    fn real_python_estimate_preserves_missing_model_error() {
        let manager = PythonSidecarManager::new();
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python-engine")
            .join("tests")
            .join("test_person.jfif");
        let missing_model_path = temp_marker("missing-model.task");
        let _ = fs::remove_file(&missing_model_path);

        let error = manager
            .estimate(EstimatePoseRequest {
                backend: "mediapipe".into(),
                image_path,
                model_path: Some(missing_model_path),
            })
            .unwrap_err();
        assert_eq!(error.kind, "python_protocol_error");
        assert_eq!(error.code.as_deref(), Some("model_asset_not_found"));
        assert_eq!(manager.ping().unwrap().status, "ready");
        manager.shutdown().unwrap();
    }

    impl PythonSidecarManager {
        fn terminate_for_test(manager: &Self) {
            if let Ok(mut inner) = manager.inner.lock() {
                Self::terminate_process(&mut inner);
            }
        }
    }
}
