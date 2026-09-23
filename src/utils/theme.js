export const APPEARANCE_STORAGE_KEY = "hpe.appearance";
export const APPEARANCE_OPTIONS = Object.freeze(["system", "light", "dark"]);

export function normalizeAppearance(value) {
  return APPEARANCE_OPTIONS.includes(value) ? value : "system";
}

export function readAppearancePreference(storage) {
  try {
    return normalizeAppearance(storage?.getItem(APPEARANCE_STORAGE_KEY));
  } catch {
    return "system";
  }
}

export function resolveTheme(appearance, systemPrefersDark) {
  const normalized = normalizeAppearance(appearance);
  return normalized === "system"
    ? systemPrefersDark ? "dark" : "light"
    : normalized;
}

export function persistAppearance(storage, appearance) {
  try {
    storage?.setItem(APPEARANCE_STORAGE_KEY, normalizeAppearance(appearance));
  } catch {
    // A denied local-storage write must not prevent theme switching.
  }
}
