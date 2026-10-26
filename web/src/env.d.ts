// Narrows Vite's catch-all `import.meta.env` typing to the variables this app reads.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
