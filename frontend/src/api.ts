import type { CameraInfo, Zone } from './types'

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = `${res.status}`
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(`${method} ${url}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  ack: (id: number) => req('POST', `/api/alerts/${id}/ack`),
  ackAll: () => req('POST', '/api/alerts/ack'),
  saveZones: (zones: Zone[]) =>
    req<Zone[]>(
      'PUT',
      '/api/zones',
      zones.map(({ id, name, type, enabled, points }) => ({ id, name, type, enabled, points })),
    ),
  resetWorkflow: () => req('POST', '/api/workflow/reset'),
  setSource: (kind: 'webcam' | 'file', value: string) => req('POST', '/api/source', { kind, value }),
  rescanCameras: () => req<CameraInfo[]>('POST', '/api/cameras/rescan'),
  videos: () => req<string[]>('GET', '/api/videos'),
  config: () => req<{ voice?: { phrases?: Record<string, string> } }>('GET', '/api/config'),
  control: (action: 'pause' | 'resume') => req('POST', '/api/control', { action }),
  voice: (enabled: boolean) => req('PUT', '/api/voice', { enabled }),
  voiceTest: () => req<{ queued: boolean; available: boolean; error: string }>('POST', '/api/voice/test'),
  upload: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('/api/upload', { method: 'POST', body: fd })
    if (!res.ok) throw new Error('Upload failed')
    return res.json() as Promise<{ source: string }>
  },
}
