# Version Control Rules — Cornell Semiconductor Project

## Architecture

This project uses a **three-channel workflow**:

| Channel | Purpose | Description |
|---------|---------|-------------|
| **GitHub** | LaTeX source, code, version control | Canonical repository. Ownership transfers to Cornell upon completion. |
| **Overleaf** (synced to GitHub) | Online LaTeX editing, compilation testing | Synced to `main` branch for collaborative editing. |
| **Google Drive** | Reference materials, style guides, raw data | Supplementary materials not tracked in git. |

## Critical Overleaf-GitHub Constraints

Overleaf's GitHub sync has hard limitations that govern the workflow:

1. **Overleaf only sees the `main` branch.** It cannot push to or pull from feature branches.
2. **Sync is manual**, not automatic. Someone must explicitly push/pull in the Overleaf UI.
3. **Merge conflicts** occur when both Overleaf and GitHub edit the same lines. When this happens, Overleaf creates a dated branch that must be merged manually.
4. **Track changes and comments** in Overleaf can be lost during GitHub pulls.
5. **No Git LFS support.** Large binary files must stay out of the repo.
6. **Recommended limits:** <100 files per commit, <100MB total project size.

## Branching Strategy

This project uses **trunk-based development**:

```text
main (trunk)                    ← Overleaf syncs here. Always deployable.
  └── feature/<topic>              ← Short-lived feature branches
```

**Rules:**

- `main` must always compile in Overleaf.
- Feature branches are created locally, worked on, then merged to `main` via fast-forward or squash merge.
- Feature branches are short-lived (hours to days, not weeks).
- Delete feature branches after merging.

## Tagging Convention

Tags mark immutable snapshots for milestones and safety:

| Tag Pattern | Purpose | Example |
|------------|---------|---------|
| `baseline-*` | Pre-edit safety snapshots | `baseline-v1` |
| `delivery-m{N}-YYYY-MM-DD` | Milestone deliveries to Cornell | `delivery-m1-m3-2026-03-01` |
| `review-*` | Pre-review checkpoints | `review-cornell-branding` |

**To revert to any tag:**

```bash
git checkout tags/<tag-name>           # View the snapshot
git checkout -b recovery/<tag-name>    # Create a branch from it
```

## Commit Message Convention

Format: `[scope] description`

Scopes:

- `[report]` — LaTeX content (chapters, frontmatter, appendices)
- `[brand]` — Cornell branding (template, colors, typography)
- `[code]` — Python pipeline code
- `[config]` — Configuration files, codebooks
- `[docs]` — Documentation, READMEs
- `[fix]` — Bug fixes
- `[meta]` — Git config, CI, project management

Examples:

```text
[report] Update cover page with ANAG authorship and Cornell TPI
[brand] Apply Carnelian header bar and Freight typography to report_main.tex
[report] Polish Chapter 1 transitions and copy-edit
[fix] Correct figure path for fig_1.3_lithography.png
```

## Workflow: Local Edits

1. **Before starting work:**

   ```bash
   git checkout main
   git pull origin main          # Get any Overleaf changes
   git checkout -b feature/<topic>  # Create feature branch
   ```

2. **While working:**
   - Make atomic commits with clear messages.
   - One logical change per commit (don't mix branding + content edits).
   - Test compilation if possible (or verify in Overleaf after merge).

3. **When ready to deliver:**

   ```bash
   git checkout main
   git pull origin main          # Catch any new Overleaf changes
   git merge feature/<topic>        # Merge feature branch
   git push origin main          # Push to GitHub → Overleaf pulls
   git branch -d feature/<topic>    # Clean up feature branch
   ```

4. **After milestone delivery:**

   ```bash
   git tag -a "delivery-m1-m3-2026-03-01" -m "M1+M2+M3 delivery to Cornell"
   git push origin "delivery-m1-m3-2026-03-01"
   ```

## Conflict Prevention

- **Never edit the same file simultaneously** in Overleaf and locally.
- **Always pull before pushing** to catch Overleaf changes.
- **Coordinate major edits** — avoid editing the same chapter in multiple environments.
- **Chapters are in separate files** (`01_introduction.tex`, `02_methods_linkpred.tex`, etc.), which naturally isolates edits.

## Recovery Procedures

**If a merge conflict occurs:**

```bash
git fetch origin
git merge origin/main           # Shows conflicts
# Manually resolve in each file
git add <resolved-files>
git commit -m "[fix] Resolve merge conflict in <file>"
git push origin main
```

**If Overleaf creates a conflict branch:**

```bash
git fetch origin
git branch -a                   # Look for overleaf-YYYY-MM-DD-* branch
git merge origin/overleaf-*     # Merge Overleaf's version
# Resolve conflicts, commit, push
```

## File Organization Rules

- **One file per deliverable** — update in place, git handles history.
- **No versioned copies** (no `_v1`, `_v2`, `_DRAFT`, `_FINAL`).
- **Binary files** (PDFs, PNGs) should be minimal — use source (LaTeX, Python) as the deliverable.
- **Large data files** stay in Google Drive, not in git.

## Known Issues

- **HPO scripts use `SUPPLYCHAIN_ROOT` for the working directory.** All 5 HPO orchestrator scripts (`src/tgnn/`, `src/graphsage/`, `src/n2v_temporal/`) resolve the project root via `os.environ.get("SUPPLYCHAIN_ROOT", str(Path(__file__).resolve().parents[2]))`. The fallback auto-detects the project root from the script's own path, so no manual configuration is needed unless running in an unusual environment.

- **GitHub repo:** `https://github.com/kbsimms/cornell-semiconductor-project.git`. **Google Drive:** `silicon-backbone:cornell-semiconductor-project` (rclone remote `silicon-backbone`, shared drive `silicon-backbone-PO1684547`).
