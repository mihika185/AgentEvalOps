const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    const message = await response.text();

    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}