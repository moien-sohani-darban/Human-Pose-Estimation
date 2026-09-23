import { useEffect, useRef } from "react";
import packageMetadata from "../../package.json";
import Icon from "./Icon";

const SettingsModal = ({
  open,
  appearance,
  onAppearanceChange,
  onClose
}) => {
  const closeRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        className="settings-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
      >
        <header className="settings-modal-header">
          <span id="settings-modal-title" className="settings-modal-header-title">Settings</span>
          <button
            className="icon-button"
            ref={closeRef}
            type="button"
            onClick={onClose}
          >
            <Icon name="close" size={12} />
          </button>
        </header>

        <div className="settings-content">
          <div className="settings-section">
            <span className="settings-section-title">Appearance</span>

            <div className="appearance-options">
              {["system", "light", "dark"].map((option) => (
                <div className={`appearance-option ${appearance === option ? "selected" : ""}`} key={option}>
                  <input
                    type="radio"
                    name="appearance"
                    value={option}
                    checked={appearance === option}
                    onChange={() => onAppearanceChange(option)}
                  />

                  {option[0].toUpperCase() + option.slice(1)}
                </div>
              ))}
            </div>

            <span className="settings-section-subtitle">{appearance === "system" ? "System follows your operating system appearance." : `${appearance[0].toUpperCase() + appearance.slice(1)} appearance is active.`}</span>
          </div>

          <div className="settings-section">
            <span className="settings-section-title">Application</span>

            <div className="application-info">
              <span className="application-info-texts version">Version {packageMetadata.version}</span>
              <span className="application-info-texts name">Human Pose Estimation</span>
              <span className="application-info-texts resources">React · Tauri · Rust · Python</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
