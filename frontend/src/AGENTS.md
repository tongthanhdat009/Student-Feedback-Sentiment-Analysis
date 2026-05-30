<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# src

## Purpose
Frontend application source code. React SPA with dark theme dashboard for managing Kaggle accounts, notebooks, and jobs.

## Key Files
| File | Description |
|------|-------------|
| `main.tsx` | React entrypoint — mounts App to DOM |
| `App.tsx` | Root component with sidebar navigation (dashboard/accounts/notebooks) |
| `index.css` | Tailwind directives, dark theme CSS custom properties, component classes (card, btn, input, badge, sidebar) |
| `vite-env.d.ts` | Vite client type declarations |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `api/` | API client and typed endpoint functions (see `api/AGENTS.md`) |
| `components/` | Reusable UI components (see `components/AGENTS.md`) |
| `routes/` | Page-level route components (see `routes/AGENTS.md`) |
| `types/` | TypeScript type definitions (see `types/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- React + Vite + TypeScript strict mode
- Tailwind CSS for styling with custom dark theme via CSS variables
- No routing library — simple state-based page switching in App.tsx
- Geist font family (sans + mono)

### Common Patterns
- API calls via `kaggleApi` object (kaggleApi.ts)
- Page components in `routes/` — Dashboard, AccountsPage, NotebooksPage
- Component UI in `components/kaggle/`

## Dependencies

### Internal
- `backend/` — REST API consumed via fetch

<!-- MANUAL: -->
