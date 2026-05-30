<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# routes

## Purpose
Page-level React components rendered in the main content area of the SPA.

## Key Files
| File | Description |
|------|-------------|
| `Dashboard.tsx` | Overview page — job stats (total/completed/running/failed), recent jobs table with download/open actions |
| `AccountsPage.tsx` | Kaggle account management — list, create, test connection, delete |
| `NotebooksPage.tsx` | Notebook inventory — list available notebooks, trigger execution |

## For AI Agents
- Pages use state-based routing via App.tsx nav
- Each page manages own data fetching lifecycle (useEffect + useState)
- Empty states handled with icons + descriptive messages

<!-- MANUAL: -->
