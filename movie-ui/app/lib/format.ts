export function formatDate(value: string) {
  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return value;
  }
}

export function formatScore(value?: number) {
  if (value === undefined || value === null) return "";
  return `${Math.round(value * 100)}% match`;
}
