# deepstream8 — repo rules

## Folder structure

Repo root holds the NVIDIA DeepStream 8 SDK tree (local build/reference copy) and
one directory per workstream. Nothing else lands at repo root.

```
deepstream8/
├── sources/  service-maker/  LICENSE.txt  README  version   ← NVIDIA SDK tree — gitignored, never commit/push
├── README.md          # repo overview + remote/SSH notes
├── CLAUDE.md          # this file
└── motion-compare/    # workstream: five motion-detection algorithms compared (see its README.md)
```

### Rules

1. **The SDK tree stays out of git.** `sources/`, `service-maker/`, `LICENSE.txt`,
   `README`, `version` at repo root are NVIDIA-proprietary
   (`LicenseRef-NvidiaProprietary`) and gitignored — never commit or push them.
   Don't modify the tree; re-pull from the NGC container / SDK installer when needed.
2. **All new work is a workstream directory at repo root** — never loose files at
   the repo root itself.
3. **Workstream layout** (`motion-compare/` is the model):

   | path | what | in git? |
   |---|---|---|
   | `motion-compare-v<N>.html` | self-contained deliverable report (clip + data inlined) | yes — it is the product |
   | `templates/` | the same pages without inlined assets (`__VIDEO__` / `__DATA__` placeholders) | yes |
   | `src/` | pipeline scripts | yes |
   | `runs/<clip>/` | per-run artifacts: `clip/`, `scores/`, `perf/`, `qa/`, `sheets/`, plus `clip.json` / `keep.json` | **no** — gitignored, exists only on this machine |
   | `data/` | raw downloaded source videos | **no** — gitignored, local only |

4. **gitignored = local-only.** Run artifacts and source videos stay on disk and
   out of git, no matter how useful they look. The deliverable HTMLs, `templates/`,
   and `src/` are tracked — the deliverables embed their clip and data inline.
5. **Don't hand-edit `runs/`** — everything under it is regenerable by the scripts
   in `src/`.

## Video QA — never in assistant-created browser tabs

The "Chrome media stack is wedged" theory is wrong and retired. Chrome and this
Mac's media stack are healthy. What stalls video is **tabs created by the Claude-in-Chrome
extension** (managed tab group, debugger-attached): any `<video>` there — data-URI, remote
mp4, WebM, even corrupt input — receives zero bytes and hangs silently (`loadstart` →
`stalled@3s` → nothing, no error). The same page in a normal user tab plays instantly.

Rule: verify video playback in a **user tab** — ask the user to open the URL, or ask
before navigating one of their tabs (navigation alone doesn't attach the debugger).
A stuck "DECODING CLIP…" spinner in an assistant-created tab is this artifact, not a
report bug. Outside the browser, AVFoundation frame dumps remain the fallback evidence.
