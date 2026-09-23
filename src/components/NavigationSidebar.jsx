import Icon from "./Icon";

const modes = [
  { id: "image", label: "Image", icon: "image" },
  { id: "video", label: "Video", icon: "video" },
  { id: "webcam", label: "Live Camera", icon: "webcam" },
];

const NavigationSidebar = ({
  inputMode,
  engineStatus,
  onModeChange,
  onModelSetup,
  onPreferences
}) => {
  const connected = engineStatus === "ready";
  return (
    <aside className="navigation-sidebar">
      <div className="nav-group-top">
        <div className="nav-group">
          <span className="nav-group-label">Workspace</span>

          {modes.map((mode) => (
            <button
              className={`nav-item ${inputMode === mode.id ? "active" : ""}`}
              key={mode.id}
              type="button"
              onClick={() => onModeChange(mode.id)}
              aria-current={inputMode === mode.id ? "page" : undefined}
            >
              <Icon name={mode.icon} size={16} />
              <span className="nav-item-label">{mode.label}</span>
            </button>
          ))}
        </div>

        <div className="nav-group tools-group">
          <span className="nav-group-label">Tools</span>

          <button
            className="nav-item"
            type="button"
            onClick={onPreferences}
          >
            <Icon name="preferences" size={16} />
            <span className="nav-item-label">Preferences</span>
          </button>
        </div>
      </div>

      <div className={`engine-card ${connected ? "connected" : ""}`}>
        <i />

        <div className="engine-card-content">
          <span className="engine-card-title">Python Engine</span>

          {connected ? (
            <div className="engine-card-status-section">
              <span className="engine-card-connection">Connected</span>
              <span className="engine-card-status-section-text">·</span>
              <span className="engine-card-status-section-text">Local</span>
            </div>
          ) : engineStatus === "loading" ?
            <div className="engine-card-status-section">
              <span className="engine-card-status-section-text">Connecting…</span>:
            </div> :
            <div className="engine-card-status-section">
              <span className="engine-card-status-section-text">Unavailable</span>
            </div>
          }
        </div>
      </div>
    </aside>
  );
}

export default NavigationSidebar;
