import test from "node:test";
import assert from "node:assert/strict";
import {
  APPEARANCE_STORAGE_KEY,
  normalizeAppearance,
  persistAppearance,
  readAppearancePreference,
  resolveTheme,
} from "./theme.js";

test("appearance normalization and system resolution are deterministic", () => {
  assert.equal(normalizeAppearance("light"), "light");
  assert.equal(normalizeAppearance("dark"), "dark");
  assert.equal(normalizeAppearance("unexpected"), "system");
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
  assert.equal(resolveTheme("light", true), "light");
});

test("appearance preference reads and writes the documented local key", () => {
  const values = new Map([[APPEARANCE_STORAGE_KEY, "dark"]]);
  const storage = {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, value); },
  };
  assert.equal(readAppearancePreference(storage), "dark");
  persistAppearance(storage, "light");
  assert.equal(values.get(APPEARANCE_STORAGE_KEY), "light");
});

test("storage access failures safely fall back to system", () => {
  const storage = {
    getItem() { throw new Error("blocked"); },
    setItem() { throw new Error("blocked"); },
  };
  assert.equal(readAppearancePreference(storage), "system");
  assert.doesNotThrow(() => persistAppearance(storage, "dark"));
});
