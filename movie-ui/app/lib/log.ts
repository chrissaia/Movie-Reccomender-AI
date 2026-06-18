export function logHandledError(context: string, err: unknown) {
  const message = err instanceof Error ? err.message : String(err);
  console.warn(`${context}: ${message}`);
}
