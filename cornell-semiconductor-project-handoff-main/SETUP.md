# Cornell Semiconductor Project - Setup Guide

This guide will help you set up your local environment to work with the Cornell Semiconductor Supply Chain project.

## Prerequisites

- Python 3.10+
- Git
- Conda (recommended for environment management)
- rclone (optional — for command-line data sync from Google Drive)

## 1. Getting Started — Choose Your Setup Path

The complete project is delivered as a single Google Drive folder containing the source code, report, and all large data files (660 GB+). The source code and report are also available on GitHub with full version history.

There are two ways to set up your local environment. Both produce the same working directory; the only difference is whether you get git history.

### Path A: Full Google Drive Download (simplest)

Download the entire delivery folder from Google Drive. This gives you everything — code, report, data, models, and results — in one step.

**Option 1: Google Drive for Desktop (easiest)**
Install [Google Drive for Desktop](https://www.google.com/drive/download/), sign in with your Cornell account, and navigate to the shared delivery folder. Files stream on demand or can be mirrored locally.

**Option 2: Browser download**
Open the shared delivery folder in your browser and download the directories you need. For 660 GB+ this may require downloading in parts.

**Option 3: rclone (command-line)**

```bash
rclone sync "silicon-backbone:cornell-semiconductor-project" /path/to/local/cornell-semiconductor-project \
    --progress \
    --transfers 4
```

See [Appendix: rclone Setup](#appendix-rclone-setup) below if you have not configured rclone before.

**Result:** A complete working directory. No git history. Suitable for reviewing results, running analysis, or verifying outputs.

### Path B: GitHub Clone + Selective Data Sync (recommended for development)

Clone the repository from GitHub for full commit history, then sync only the large data directories from Google Drive.

**Step 1: Clone the repository**

```bash
git clone https://github.com/kbsimms/cornell-semiconductor-project.git
cd cornell-semiconductor-project
```

**Step 2: Sync data directories from Google Drive**

The following directories are too large for GitHub and must be synced from the delivery Drive folder. Using rclone:

```bash
# Sync each data directory into your cloned repo
rclone sync "silicon-backbone:cornell-semiconductor-project/data" ./data --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/results" ./results --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/artifacts" ./artifacts --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/predictions" ./predictions --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/logs" ./logs --progress
```

Alternatively, download these 5 directories from Google Drive via browser or Drive for Desktop and place them in the repository root.

**Do NOT** `rclone sync` the entire Drive folder into a git clone — this will overwrite the `.git/` directory and destroy your commit history.

**Result:** Identical files to Path A, plus full git history (branches, tags, commit log). Recommended if you plan to modify code, track changes, or contribute.

## 2. Set Up Python Environment

Create and activate the conda environment:

```bash
# For GPU support (recommended — required for model training)
conda env create -f environment_gpu.yml
conda activate supplychain_env_gpu

# OR for CPU-only (sufficient for analysis scripts and report compilation)
conda env create -f environment_cpu.yml
conda activate supplychain_env_cpu
```

## 3. Set Up FactSet Credentials (if needed)

The data processing pipeline can query FactSet's Supply Chain Relationships database via SQL. This requires database credentials stored in a `.env` file.

**If you need to run `src/db_client/` or core data processing pipeline steps:**

```bash
cp .env.example .env
# Edit .env and fill in your FactSet credentials:
#   DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD
```

A working `.env` file is included in the Google Drive delivery folder. You can copy it directly into the repository root.

**If you are only running analysis scripts or reviewing results**, you do not need FactSet credentials — the processed datasets in `data/processed/` and `artifacts/` are self-contained.

Note: The `database` section in `configs/global_config.yaml` is reference documentation only. The actual database connection uses environment variables from `.env` via `src/db_client/connection.py`.

## 4. Verify Setup

1. **Check repository structure:**

```bash
ls -la
# Should see: src/, configs/, docs/, data/, artifacts/, results/, report/, etc.
```

1. **Check data directories:**

```bash
ls -lh data/ artifacts/ results/ 2>/dev/null | head -5
# Should see your synced data files
```

1. **Test Python import:**

```bash
python -c "import torch; import pandas; print('Setup successful!')"
```

## 5. Working with the Repository

### Making Changes

1. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

1. Make your changes and commit:

```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

1. Create a Pull Request on GitHub for review.

### Syncing Code Updates (Path B only)

```bash
git pull origin main
```

## 6. Important Notes

### Data Access

- **Google Drive access:** Access to the Google Drive delivery folder will be provisioned by the Cornell project lead upon handoff. Contact them to confirm your access.
- **Large files:** The `.gitignore` excludes large directories (data/, artifacts/, results/, predictions/, logs/) from git. These are only available via the Google Drive delivery folder.
- **Data is frozen:** The delivered data files represent the final state of the project. There is no ongoing sync — what you download is the complete dataset.

### Disk Space

- The full dataset requires approximately **700 GB** of free disk space.
- If disk space is limited, you can download only the directories you need. The minimum for running analysis scripts is `data/processed/` (~12 GB) and `artifacts/` (~25 GB). The `results/` directory (590 GB of model score parquets) is only needed to reproduce evaluation tables and figures.

### Permissions

- **GitHub:** Access level is set by the Cornell project lead. Researchers typically have write access (create branches, push, create PRs).
- **Google Drive:** Access level is set by the Cornell project lead upon handoff.

## 7. Troubleshooting

### rclone Permission Errors

If you get permission errors when accessing Google Drive:

1. Verify you've been granted access to the shared folder
2. Re-authenticate: `rclone config reconnect gdrive`
3. Check folder name: `rclone lsd gdrive:` to see available folders

### Sync Issues

- If sync is slow, try syncing during off-peak hours
- Use `--transfers` flag to limit concurrent transfers: `rclone sync ... --transfers 4`
- Check your internet connection and Google Drive API quotas

### Python Environment Issues

- Make sure you've activated the conda environment: `conda activate supplychain_env_gpu`
- If packages are missing, reinstall: `conda env update -f environment_gpu.yml`

## 8. Report Compilation

The report is compiled using pdfLaTeX + Biber. The Overleaf project is the canonical compilation environment; access will be provisioned by the Cornell project lead.

1. The Overleaf project root must be the **repository root** (the parent of `report/`), not `report/` itself — all `\include` and `\includegraphics` paths are relative to this level.
2. Set the compiler to **pdfLaTeX** and bibliography tool to **Biber**.
3. Compile `report/report_main.tex` as the main document.
4. All six sections and three appendices are active in `report_main.tex` and should compile without modification.

If compiling locally, ensure `newunicodechar.sty` and all packages listed in the preamble are installed in your TeX distribution.

## 9. Getting Help

- **Repository Issues:** Create an issue on GitHub
- **Data Access:** Contact the Cornell Brooks Technology Policy Institute project lead
- **Technical Questions:** Check the README.md or project documentation in `docs/`

---

## Appendix: rclone Setup

If you have not used rclone before, follow these steps to configure it for Google Drive access.

### Install rclone

**On Ubuntu/Debian:**

```bash
sudo apt-get install rclone
```

**On macOS:**

```bash
brew install rclone
```

**Or download from:** <https://rclone.org/install/>

### Configure rclone for Google Drive

```bash
rclone config
```

Follow the prompts:

- **n** (create new remote)
- Name it: `gdrive` (or your preferred name)
- Select: **drive** (Google Drive)
- Leave client_id and client_secret blank (press Enter)
- Select: **1** (Full access to all files)
- Leave service_account_file blank (press Enter)
- **n** (not advanced config)
- **y** (auto config — this will open a browser)
- Authenticate with your Google account
- **y** (confirm remote is correct)
- **q** (quit config)

### Verify Access

```bash
rclone lsd "silicon-backbone:cornell-semiconductor-project"
```

You should see directories including `artifacts/`, `data/`, `results/`, `src/`, and `report/`. If you get a permission error, contact the Cornell project lead to confirm your Google Drive access.

---

**Last Updated:** April 2026
**Originally developed by:** Asymmetric Network Advisory Group LLC for Cornell University
**Contact:** Create an issue on GitHub or contact the Cornell project lead
