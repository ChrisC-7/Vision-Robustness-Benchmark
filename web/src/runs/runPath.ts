/**
 * The app route for a run: /runs/ followed by its run_dir, one encoded URL
 * segment per path segment, e.g. /runs/runs/multiseed/experiment1/simple_s42.
 * The route is a splat, so the run_dir comes back intact as params['*'].
 */
export function runPath(runDir: string): string {
  return `/runs/${runDir.split('/').map(encodeURIComponent).join('/')}`
}

/** The experiment a run belongs to: the folder its run_dir sits in. */
export function experimentName(runDir: string): string | undefined {
  const parts = runDir.split('/')
  return parts.length > 1 ? parts[parts.length - 2] : undefined
}
