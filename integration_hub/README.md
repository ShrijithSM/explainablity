# Integration Hub

This folder is a separate integration workspace to connect the Chronoscope codebase with a user interface.

## Structure

- frontend: UI application and assets
- backend: API and adapter layer that talks to Chronoscope modules
- shared: contracts and shared types between frontend and backend
- configs: environment and runtime configuration files
- scripts: utility scripts for local orchestration
- docs: architecture and integration notes

## Next Integration Steps

1. Define shared request and response contracts in shared/contracts.
2. Add a backend adapter in backend/adapters/chronoscope to call Chronoscope analysis flows.
3. Expose minimal API endpoints in backend/api.
4. Wire frontend views in frontend/src to backend endpoints.
5. Add launch scripts in scripts for local end-to-end runs.
