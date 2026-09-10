/** Build-time limits must remain usable HTML maxLength values. */
export function positiveIntegerSetting(value: unknown, fallback: number) {
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback
}
