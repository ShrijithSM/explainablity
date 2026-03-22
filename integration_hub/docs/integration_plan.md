# Integration Plan

## Phase 1: Folder and boundaries

- Keep integration work isolated under integration_hub.
- Reuse Chronoscope logic by importing from the main project package.
- Avoid changing existing experiment scripts during initial integration.

## Phase 2: API contract first

- Define payload format for prompt, model settings, and analysis options.
- Define response shape for metrics, causality summaries, and report links.

## Phase 3: Incremental wiring

- Build backend adapter, then API route.
- Build frontend connection to route.
- Add basic health check and one analysis flow.
