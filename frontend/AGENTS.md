<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# frontend

## Purpose
React + Vite + Tailwind CSS single-page application for the Kaggle Notebook Manager. Dark-themed dashboard for managing accounts, triggering notebook runs, and monitoring job status.

## Key Files
| File | Description |
|------|-------------|
| `package.json` | Frontend dependencies (React, Vite, TypeScript, Tailwind, lucide-react) |
| `tsconfig.json` | TypeScript strict mode config |
| `tailwind.config.js` | Tailwind CSS content paths |
| `postcss.config.js` | PostCSS config for Tailwind |
| `eslint.config.js` | ESLint flat config |
| `index.html` | Vite entry HTML |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | Application source code (see `src/AGENTS.md`) |
| `public/` | Static assets |
| `dist/` | Production build output (gitignored) |

## For AI Agents

### Working In This Directory
- `npm run dev` for dev server (Vite)
- `npm run build` for production build
- CSS uses Tailwind `@apply` + CSS custom properties for dark theme
- Geist / Geist Mono font family

### Testing Requirements
- Manual visual testing in browser
- TypeScript validation: `tsc --noEmit`

### Common Patterns
- Pages in `src/routes/`, components in `src/components/kaggle/`
- API client in `src/api/client.ts`, typed endpoints in `src/api/kaggleApi.ts`
- Types defined in `src/types/kaggle.ts`

## Dependencies

### Internal
- `backend/` — FastAPI REST API consumed by frontend

### External
- React 19 — UI library
- Vite 6 — Build tool
- Tailwind CSS 3 — Utility-first CSS
- lucide-react — Icon library

<!-- MANUAL: -->
