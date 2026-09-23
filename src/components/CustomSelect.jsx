import { useEffect, useId, useRef, useState } from "react";
import Icon from "./Icon";

export default function CustomSelect({
  ariaLabel,
  value,
  options,
  onChange,
  disabled = false,
  className = "",
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const menuId = useId();
  const selectedOption = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return undefined;

    const closeOnOutsideClick = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={`custom-select ${className} ${open ? "open" : ""}`}>
      <button
        type="button"
        className="custom-select-trigger"
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="custom-select-label">{selectedOption?.label ?? "Select"}</span>
        <Icon name="chevron" className="custom-select-chevron" size={8} />
      </button>

      <div className="custom-select-menu" hidden={!open}>
        {options.map((option) => (
          <button
            className={`custom-select-option ${option.value === value ? "selected" : ""}`}
            key={option.value}
            type="button"
            disabled={option.disabled}
            onClick={() => {
              onChange(option.value);
              setOpen(false);
            }}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
