# Master Change Log — The Silicon Backbone

This log covers editorial, bibliographic, and client-readiness changes made during the final editorial review (February 2026). Full development history is available via `git log`.

**Repository:** `https://github.com/kbsimms/cornell-semiconductor-project` *(URL will update upon ownership transfer to Cornell)*

---

## Citation Audit Corrections (March 28, 2026)

Independent QA audit of all 219 citations across 6 chapters and 3 appendices. Five-agent verification swarm followed by four deep-research verification agents with 2-cycle academic-standard investigation per finding.

### Critical Fixes

- **NDAA §856 misattribution (Ch5 §5.7):** Separated statutory text ("a combination of government and commercial tools") from HASC committee report language (AI/ML). Previous text conflated the two; committee reports do not carry force of law. Verified against 5 independent legal analyses and GAO-25-107283.
- **McGillis & Reed (2026) hybrid strategy (Ch5 §5.5–5.8):** Reduced from 8 citations to 2 (lines 140, 142) with disclosure footnote noting conference paper status. Replaced 6 instances with published NIST IR 8536 and NIST IR 8419. Both authors' expertise and substantive claims independently verified.
- **NIST IR 8536 identifier (bib):** Corrected "NIST IR 8536.2" → "NIST IR 8536 (Second Public Draft)"; added 5 named authors; disclosed draft status (not finalized as of March 2026).
- **SD-26 DMSMS guidebook (bib):** Corrected author (Defense Standardization Program Office, not DAU), title, year (2023, not 2024), and URL (dead link replaced).

### Metadata Fixes (references.bib)

- DFARS 252.246-7007: year 2024 → 2023 (JAN 2023 promulgation)
- DFARS 252.246-7008: year 2024 → 2023 (JAN 2023 promulgation)
- DFARS 252.239-7018: year 2024 → 2022 (DEC 2022 promulgation)
- kaplan\_quantitative\_1981: DOI truncated — added missing trailing `.x`
- Page1999: author "Motwani, T" → "Motwani, Rajeev" (full names for all authors)
- ch5\_usc2223: title corrected to statutory title per Cornell Law Institute
- ch5\_tpp\_guidebook\_2022: dead URL (403) replaced with working DAU link

### Additions

- **NIST IR 8419** (Stouffer, Pease, Reed et al., 2022): New bib entry — blockchain-based manufacturing supply chain traceability. Bib count 218 → 219.
- **Conference provenance note** for McGillis & Reed (2026): Disclosed 12th IAA STM Conference presentation status and that proceedings are not yet publicly indexed.

### Cleanup

- Removed 2 orphan bib entries never cited in any chapter: `chen_heterogeneous_2017`, `department_of_energy_basic_2018`. Final bib count: 217.
- Footnote wording tightened: "published works of the same authors" → "published works of Reed and colleagues" (only Reed overlaps between McGillis/Reed and the NIST IRs).

## Critical Prose Fixes (March 28, 2026)

### Voice — "chapter" → "section" in prose (20 replacements)

- Ch2: 15 self-referential and cross-referential "chapter" → "section" (includes "Chapters~3 and 4" → "Sections~3 and 4")
- Ch3: 4 replacements ("The chapter proceeds", "The chapter closes", 2× "later in this chapter")
- AppA: 1 replacement ("Chapter~\ref{}" → "Section~\ref{}")
- Deep-research verification confirmed 0 prose "chapter" instances remain in Ch4, Ch5, Ch6, AppB, AppC, or executive summary

### Dissertation Language Removal

- Ch1 line 8: "The contribution of this section is structural rather than speculative" → "This section addresses structure rather than speculation"
- Ch4 line 310: "core contribution of this section" → "primary decision output of this section"

### Compound Modifier

- Ch5 line 10: "mission-critical" → "mission-critical"

### Issues Verified as Non-Issues (no fix applied)

- Ch3 alleged subject-verb disagreement ("treating…and represents"): DENIED — text is "treats…and represents", grammatically correct
- Ch5 alleged 100+ word run-on: DENIED — sentence is 40 words
- Ch4 alleged dangling modifier: optional cosmetic only, not a true dangling modifier per major style guides
**Period:** February 24–26, 2026
**Bibliography:** 178 → 217 entries (+16 new in M1–M3; +21 ch5_ policy/regulatory keys + 3 restored in M5/M6; −10 deleted in M1–M3; +1 NIST IR 8419 in citation audit; −2 orphans removed in citation audit)
**Audit scope:** Two full rounds — Round 1 (7-agent forensic audit, 55-fix plan) and Round 2 (9-agent swarm re-audit, 48-item remediation register)
**Note:** This log covers the systematic audit and remediation work. The full commit history includes additional commits for Overleaf integration, Cornell branding, Grammarly repairs, frontmatter updates, and repository setup — see `git log` for the complete record.

---

## Milestones 5 & 6 — Chapters 5 and 6 Integration (March 21, 2026)

### Ch5 & Ch6 Source Integration

- Added contractor-supplied Chapter 5 (`05_illumination_to_assurance.tex`: From Illumination to Assurance — policy recommendations) and Chapter 6 (`06_conclusion.tex`: Conclusion)
- Added `tab_5_1_implementation_roadmap.tex` (Table 5.1: implementation roadmap for policy recommendations)
- Updated `references.bib` (200 → 216 entries, +21 `ch5_` policy/regulatory keys for DFARS, DoDI, NIST, NTIA, FAR citations)
- Deleted empty stub `05_conclusion_policy.tex`

### Wiring and Cross-Reference Fixes

- Wired Ch5 and Ch6 into `report_main.tex` (previously commented out)
- Fixed 13 cross-reference label mismatches in Ch5/Ch6 (`chap:link_prediction` → `chap:link-prediction`, `chap:dod_supply_chain` → `chap:dod-supply-chain`) to match existing chapter labels
- Fixed Ch5 table `\input{}` path (`tables/` → `report/tables/`) for Overleaf compilation
- Replaced 4 hardcoded `Section~5` forward references with `\ref{chap:from_illumination_to_assurance}` in Ch1 (1×) and Ch4 (3×)
- Updated executive summary: Ch5 and Ch6 described in present tense; "five sections" → "six sections"
- Updated cover.tex: "Milestones 1 through 4" → "Milestones 1 through 6"; removed "will follow in a subsequent milestone"

### Documentation

- Updated `PROJECT_MAP.md`: LaTeX dependency tree (6 active chapters, 16 figures, 18 tables, 218 bib entries), removed commented-out stub references

### Editorial Conversion — Dissertation to Consulting Voice (March 21)

- Converted Ch5 and Ch6 from first-person academic/dissertation voice to impersonal consulting voice, matching Sections 1–4
- ~58 "Chapter" → "Section" replacements in prose text (LaTeX structural commands unchanged)
- ~46 "dissertation" → "report"/"analysis" replacements across Ch5 and Ch6
- 27 first-person pronoun eliminations ("I [verb]" → "the analysis [verb]s" or passive)
- ~15 "contribution" → "finding"/"output"/"result" reframings in Ch6
- Section heading rename: "What This Dissertation Contributes" → "What This Analysis Delivers"
- Label rename: `sec:ch6_what_this_dissertation_contributed` → `sec:ch6_what_this_analysis_delivers`
- 3 tone/jargon fixes: "identificational" → "concerns identity linkage", "deeper scholarship" → "deeper analysis", "future research agenda" → "future work"
- Executive summary rewritten: expanded Section 5/6 coverage to match Section 4 quality bar (decision-grade visibility objective, six design principles, four assurance layers, four linked findings, inference boundaries)
- Ch1 roadmap updated: corrected Ch5 description (limitations properly attributed to Ch6), added Section 6 paragraph
- Cover.tex updated: "near-final form" → "final form", removed forward-looking Executive Summary language
- 3 conversion artifacts fixed post-audit: capitalization (Ch5 line 74), subject-verb agreement (Ch5 line 126), mid-sentence capital (Ch6 line 18)

---

## Milestone 4 — Chapter 4 Integration and QA (March 2026)

### Ch4 Source Integration (March 3)

- Added contractor-supplied Chapter 4 source files (severity scenarios, vulnerability analysis)
- Wired Ch4 and Appendix C into `report_main.tex`
- Added `\usepackage{tikz}` for action matrix figure
- Fixed hyperref warning in section title (`\texorpdfstring`)

### Ch4 Restyle (March 3)

- Converted Ch4 and Appendix C from first-person academic voice to impersonal consulting voice (~150 lines)
- Normalized typography (straight quotes, cross-references)
- Verified and corrected all 6 citations (Kaplan DOI fix, Page1999 author fix, all names expanded)
- Added 8 `\label{sec:ch4_*}` labels and replaced 22 hardcoded section references with `\ref{}`
- Merged Grammarly prose improvements from Overleaf; surgically repaired 7 garbled passages

### Pipeline Orchestrator (March 3)

- Added `src/analysis/chapter4/run_pipeline.sh` — production M0→M9 orchestrator with 32 run_module calls
- Supports `--config`, `--skip-m9-2` (offline), `--skip-reporting` flags
- Pre-flight checks for Ch3 input artifacts; fail-fast (`set -euo pipefail`)

### M4 QA Swarm (March 3)

- 13-agent quality assurance audit across 7 tracks: SOW compliance, content accuracy, argumentation quality, prose (Ch1-3 and Ch4), code syntax, code-to-report traceability, documentation, LaTeX/bib integrity, security
- **Result:** 0 numerical errors in 43 verified claims; 6 critical documentation gaps, 24 major findings, 37 minor
- All critical and major findings remediated in subsequent commits

### Remediation (March 3–4)

- Fixed module contract paths (`common/` → `ch4_common.py`, `ch4_v2.yaml` → `ch4_v2_fix01.yaml`)
- Documented `run_pipeline.sh` in Ch4 README, top-level README, and PROJECT_MAP.md
- Fixed TeX engine reference in PROJECT_MAP.md (xelatex → pdflatex)
- Standardized asset paths in Ch4 and AppC (`figures/` → `report/figures/`)
- Removed 7 first-person voice instances across Ch1, Ch3, AppA
- Replaced AI vocabulary: comprehensive, crucial, notably
- Fixed `wafter` → `wafer` typo in figure label
- Fixed sentence fragment, punctuation corruption, missing commas
- Added budget-k justification, time-window robustness scope-out, P(n) and integrity scope-outs
- Fixed Python 3.14 escape sequence compatibility in `05_build_candidate_pools.py`
- Documented manually maintained report assets, script-to-report mapping, inline statistics

---

## Milestone 1 — Chapter 1: Semiconductor Primer

### Content Corrections

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Earthquake magnitude 7.2 → 7.4, added USGS source | USGS Mw 7.4 is the international standard; CWA local scale revised to 7.1, making 7.2 wrong on every scale | [`bcd2797`](https://github.com/kbsimms/cornell-semiconductor-project/commit/bcd2797) | `01_introduction.tex`, `references.bib` |
| Replaced Reuters fire article with Matsuo 2015 for 2011 Tohoku earthquake | Reuters covered a 2021 Renesas factory fire, not the 2011 earthquake shutdown | [`1fb1c51`](https://github.com/kbsimms/cornell-semiconductor-project/commit/1fb1c51) | `01_introduction.tex`, `references.bib` |
| Replaced empty stub with JEDEC JESD238 standard for HBM3 architecture | Previous entry had no author, URL, publisher, or DOI | [`1fb1c51`](https://github.com/kbsimms/cornell-semiconductor-project/commit/1fb1c51) | `01_introduction.tex`, `references.bib` |
| Replaced political science paper with Van Zant + IRDS for fab energy/materials | Williams 2003 (securitization theory) was unrelated to semiconductor manufacturing | [`1fb1c51`](https://github.com/kbsimms/cornell-semiconductor-project/commit/1fb1c51) | `01_introduction.tex`, `references.bib` |
| Replaced F-35 software depot article with CSIS semiconductor report | Heusel & Hill 2017 covered software maintenance, not semiconductor supply chains | [`308979b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/308979b) | `01_introduction.tex` |

### Citation Alignment (Wrong Source for Claim)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Removed EU Chips Act from US DoD policy parencite | EU legislation irrelevant to a sentence about US defense policy | [`9eab98d`](https://github.com/kbsimms/cornell-semiconductor-project/commit/9eab98d) | `01_introduction.tex` |
| Replaced CUDA/GPU paper with Hauck 2008 for FPGA claims | Nickolls 2008 covers GPU architecture, not FPGA pipeline latency or DSP slices | [`9eab98d`](https://github.com/kbsimms/cornell-semiconductor-project/commit/9eab98d) | `01_introduction.tex` |
| Replaced Chen 2017 roadmap intro with HIR 2024 for HBM | 4-page overview insufficient; full HIR edition covers HBM architecture | [`9eab98d`](https://github.com/kbsimms/cornell-semiconductor-project/commit/9eab98d) | `01_introduction.tex` |
| Removed Jacob 2008 from ADC/DAC and FPGA parencites | DRAM/cache textbook does not cover mixed-signal or FPGA internals | [`9eab98d`](https://github.com/kbsimms/cornell-semiconductor-project/commit/9eab98d) | `01_introduction.tex` |
| Replaced HIR 2024 with IRDS Lithography 2023 for fluorinated gas purity | Packaging roadmap does not cover front-end-of-line process chemistry | [`bcd2797`](https://github.com/kbsimms/cornell-semiconductor-project/commit/bcd2797) | `01_introduction.tex` |
| Removed DOE Basic Research 2018 from overlay metrology cite | Fundamental science report does not cover manufacturing process specs | [`bcd2797`](https://github.com/kbsimms/cornell-semiconductor-project/commit/bcd2797) | `01_introduction.tex` |
| Removed OECD from counterfeit/integrity cluster | OECD paper measures subsidies, not counterfeits | [`bcd2797`](https://github.com/kbsimms/cornell-semiconductor-project/commit/bcd2797) | `01_introduction.tex` |
| Replaced Neisser 2021 with Weste 2011 for transistor count claim | Lithography stochastics paper does not cover IC density/transistor counts | [`bcd2797`](https://github.com/kbsimms/cornell-semiconductor-project/commit/bcd2797) | `01_introduction.tex` |
| Reduced Neisser 2021 from 13 to 4 citations | Retained in lithography contexts only; removed from GaN, FPGA, GPU, memory, SWaP, packaging | [`308979b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/308979b) | `01_introduction.tex` |
| Removed OECD from DoD procurement cite | Trade distortions paper was misapplied to counterfeit acceptance processes | [`308979b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/308979b) | `01_introduction.tex` |

### New Source Supplements (No Removals)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| National Academies 2024 added at lines 24, 26, 32, 38 | Consensus study on DoD semiconductor access — strengthens core thesis claims | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c), [`81c99ed`](https://github.com/kbsimms/cornell-semiconductor-project/commit/81c99ed) | `01_introduction.tex`, `references.bib` |
| Tehranipoor 2010 (1,101 cites) added at line 253 | Foundational hardware Trojan taxonomy — strengthens insertion pathway claim | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `01_introduction.tex`, `references.bib` |
| Guin 2014 (478 cites) added at line 251 | Peer-reviewed counterfeit failure signatures — supplements White House 2021 | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `01_introduction.tex`, `references.bib` |
| USITC Japan-Korea 2020 added at line 243 | Japan specialty chemicals market shares (confirms ~70% figure) | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `01_introduction.tex`, `references.bib` |
| CSIS China Rare Earth 2025 added at line 243 | China rare earth export threat documentation | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `01_introduction.tex`, `references.bib` |
| SIA FCC Filing 2021 added at line 98 | Primary source for "1,400 process steps" claim (alongside Khan 2021) | [`c61177b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/c61177b) | `01_introduction.tex`, `references.bib` |
| DoD Securing 2022 supplemented at line 48 | Strengthens NSCAI 2021 cite for chip lead-time propagation into program timelines | [`81c99ed`](https://github.com/kbsimms/cornell-semiconductor-project/commit/81c99ed) | `01_introduction.tex` |
| Pergolizzi PGK 2010 added at line 42 | NDIA presentation documents GPS guidance and electronic fuze modes in artillery — supplements Hoehn 2021 | [`610afd1`](https://github.com/kbsimms/cornell-semiconductor-project/commit/610afd1) | `01_introduction.tex`, `references.bib` |
| NDA/confidentiality cite added at line 230 | OECD semiconductor value chain opacity — fills gap in sub-tier visibility claim | [`4ed56a2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ed56a2) | `01_introduction.tex` |
| Japan chemical export claim strengthened at line 243 | Kim 2019 co-cited alongside Yamazaki 2019 | [`80b65ab`](https://github.com/kbsimms/cornell-semiconductor-project/commit/80b65ab) | `01_introduction.tex` |

---

## Milestone 2 — Chapter 2: Link Prediction Methods

### Content Corrections

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Replaced Sculley hash key with human-readable key | `NIPS2015_86df7dcf` renders poorly in PDF bibliography | [`516b7e9`](https://github.com/kbsimms/cornell-semiconductor-project/commit/516b7e9) | `02_methods_linkpred.tex`, `references.bib` |
| Removed Cohen 2008 from temporal split methodology cite | Stock return prediction paper was tangential; Bekker 2020 alone covers PU temporal split | [`308979b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/308979b) | `02_methods_linkpred.tex` |

### New Source Supplements (No Removals)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Xu TGAT 2020 (691 cites) added at line 58 | Provides 70/15/15 temporal split precedent in graph attention networks | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `02_methods_linkpred.tex`, `references.bib` |
| Robinson 2021 (415 cites) added at line 66 | Hard negative sampling best practices for contrastive learning | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `02_methods_linkpred.tex`, `references.bib` |
| Hamilton 2016 added at line 232 | Orthogonal Procrustes alignment for diachronic embeddings | [`73e4d4c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/73e4d4c) | `02_methods_linkpred.tex`, `references.bib` |
| Elkan 2008 (948 cites) supplemented at line 33 | Foundational PU precision lower-bound theorem — supplements Gopal 2021 | [`a8d5460`](https://github.com/kbsimms/cornell-semiconductor-project/commit/a8d5460) | `02_methods_linkpred.tex` |
| Yang 2020 (182 cites) added at line 204 | Theoretical basis for degree-biased negative sampling in graph representation learning | [`8f9b0cc`](https://github.com/kbsimms/cornell-semiconductor-project/commit/8f9b0cc) | `02_methods_linkpred.tex`, `references.bib` |
| Goodfellow 2016 added at line 58 | Standard reference for 70/15/15 train/val/test split convention | [`4ed56a2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ed56a2) | `02_methods_linkpred.tex` |
| Holme 2012 added at line 228 | Temporal networks review covering time-window aggregation granularity | [`4ed56a2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ed56a2) | `02_methods_linkpred.tex` |

### Missing Formalisms Cited

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Brin & Page 1998 added for PageRank damping factor 0.85 | The paper that introduced PageRank — fills uncited formalism | [`80b65ab`](https://github.com/kbsimms/cornell-semiconductor-project/commit/80b65ab) | `02_methods_linkpred.tex`, `references.bib` |

---

## Milestone 3 — Chapter 3: DoD Network Construction

### Content Corrections

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| TSMC earthquake citation replaced | Original cited a Jan 2025 revenue report; replaced with Reuters Apr 2024 Hualien quake article | [`516b7e9`](https://github.com/kbsimms/cornell-semiconductor-project/commit/516b7e9) | `03_dod_build_shipping_semiconductor.tex`, `references.bib` |

### Missing Citations Added

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Craighead 2007 added at line 190 | Common-mode vulnerability framework for dual-supplier retention threshold | [`4ed56a2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ed56a2) | `03_dod_build_shipping_semiconductor.tex` |
| FactSet 2025 added at line 275 | Primary data source methodology for shipping coverage asymmetry claim | [`4ed56a2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ed56a2) | `03_dod_build_shipping_semiconductor.tex` |
| Kivelä 2014 (2,845 cites) added for directed multiplex graph formalism | Foundational multilayer network review — fills uncited formalism in §3.5 | [`80b65ab`](https://github.com/kbsimms/cornell-semiconductor-project/commit/80b65ab) | `03_dod_build_shipping_semiconductor.tex`, `references.bib` |
| OMB SIC Manual 1987 added for SIC 3674 classification | Primary source for the SIC code used as semiconductor industry boundary | [`80b65ab`](https://github.com/kbsimms/cornell-semiconductor-project/commit/80b65ab) | `03_dod_build_shipping_semiconductor.tex`, `references.bib` |
| GAO 2025 added for USAspending sub-tier visibility gap | Defense industrial base sub-tier dependence documentation | [`80b65ab`](https://github.com/kbsimms/cornell-semiconductor-project/commit/80b65ab) | `03_dod_build_shipping_semiconductor.tex` |

---

## Cross-Milestone — Bibliography Maintenance

### Duplicate/Orphan Cleanup

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Deleted 5 duplicate orphan entries | `rokach_evaluating_2011`, `noauthor_japanese_2021`, `noauthor_tsmc_2025`, `noauthor_data_nodate`, `us_department_of_the_treasury_data_nodate` — all duplicates of existing entries | [`69d8503`](https://github.com/kbsimms/cornell-semiconductor-project/commit/69d8503) | `references.bib` |
| Deleted 2 orphaned entries post-alignment | `heusel_hill_2017` and `cohen_economic_2008` — zero .tex references after Batch 8 fixes | [`e76051f`](https://github.com/kbsimms/cornell-semiconductor-project/commit/e76051f) | `references.bib` |
| Deleted 3 orphaned entries post-replacement | `williams_words_2003`, `noauthor_high_2025`, `reuters_japanese_2021` — replaced by correct sources in Batch 1 | [`1fb1c51`](https://github.com/kbsimms/cornell-semiconductor-project/commit/1fb1c51) | `references.bib` |

### Dead URL Fixes

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Fixed 4 dead URLs to verified alternatives | IRDS Lithography (404→PDF), IRDS Metrology (404→PDF), IEEE EPS HIR 2024 (404→new scheme), SEMI HIR 2021 (404→new scheme) | [`6538c67`](https://github.com/kbsimms/cornell-semiconductor-project/commit/6538c67) | `references.bib` |

### Metadata Corrections (Round 1)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Corrected metadata for 15 entries | Entry types, co-authors, DOIs, booktitles, publisher/address swaps, author order — see commit for full list | [`0fe6814`](https://github.com/kbsimms/cornell-semiconductor-project/commit/0fe6814) | `references.bib` |
| Added missing URLs and fields to 6 entries | IRDS Factory Integration, NSTC, GAO, ISO, DOE, IJCAI — all URLs verified live | [`2c2b8da`](https://github.com/kbsimms/cornell-semiconductor-project/commit/2c2b8da) | `references.bib` |

### Metadata Corrections (Round 2 — Bibliographic)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| GAO-12-375 title corrected, institution fields added to GAO-12/16, du Plessis pages | Official GAO title was truncated; institution fields missing | [`4ef8f84`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4ef8f84) | `references.bib` |
| 3 DOIs added, GraphSAGE pages, Patterson co-author | Fawcett, Fellegi, Kanarik missing DOIs; Hamilton 2017 missing pages; Hennessy missing Patterson | [`8b456f2`](https://github.com/kbsimms/cornell-semiconductor-project/commit/8b456f2) | `references.bib` |
| Lopez note fixed, 4 DOIs added, Zotero artifacts cleaned | "Dept of War" anachronistic for 2023 article; Lu, Nickolls, Puurunen, Real missing DOIs; Neamen/Pettit had export artifacts | [`45a7305`](https://github.com/kbsimms/cornell-semiconductor-project/commit/45a7305) | `references.bib` |
| 2 co-authors added, CSIS entry type fixed, Winkler URL, OSTI artifacts | Sharma missing Bhattacharya, Taur missing Ning; Shivakumar was @article not @techreport; OSTI "None" artifacts | [`299f85c`](https://github.com/kbsimms/cornell-semiconductor-project/commit/299f85c) | `references.bib` |

### Formatting Fixes (Round 2)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| 7 formatting corrections | OECD/DoD institution fields, Kuon booktitle, Pareja @inproceedings, Vergun howpublished, Chen author initials, White House brace protection | [`d332fd9`](https://github.com/kbsimms/cornell-semiconductor-project/commit/d332fd9) | `references.bib` |

---

## Client-Readiness Audit — Track 3: Code Quality

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Fix copy-paste docstring (M1-4) | HPO script referenced wrong module name (`00_tgnn_hpo...` → `00_n2v_temporal_hpo...`) | [`d6bf2c0`](https://github.com/kbsimms/cornell-semiconductor-project/commit/d6bf2c0) | `n2v_temporal/00_n2v_temporal_hpo_three_phase.py` |
| Fix mutable default argument (M4-3) | `columns: List[str] = [...]` shared mutable list across instantiations; replaced with `Optional[List[str]] = None` pattern | [`d6bf2c0`](https://github.com/kbsimms/cornell-semiconductor-project/commit/d6bf2c0) | `heuristics/run_heuristics_eval.py` |
| Rename `XXX` → `NNN` in error message (Mi1-2) | Avoided false positives from automated TODO/XXX scanners | [`d6bf2c0`](https://github.com/kbsimms/cornell-semiconductor-project/commit/d6bf2c0) | `n2v_temporal/run_n2v_temporal_eval.py` |
| Standardize seed initialization (M5-2, M5-3) | Added `random.seed()` + `torch.cuda.manual_seed_all()` to TGNN and n2v_temporal eval; matches twotower/graphsage pattern | [`368c272`](https://github.com/kbsimms/cornell-semiconductor-project/commit/368c272) | `tgnn/run_tgnn_eval.py`, `n2v_temporal/run_n2v_temporal_eval.py` |
| Add missing dependencies (M3-1, M3-2) | `fuzzywuzzy`, `python-Levenshtein`, `plotly` missing from both env files; fresh install would fail | [`181fcf8`](https://github.com/kbsimms/cornell-semiconductor-project/commit/181fcf8) | `environment_cpu.yml`, `environment_gpu.yml` |
| Remove EvolveGCN dead code (M1-3) | `global_config.yaml` states model removed; dead code path silently produced empty results instead of erroring | [`209ebe3`](https://github.com/kbsimms/cornell-semiconductor-project/commit/209ebe3) | `scorecard_common.py` |
| Create `.env.example` (Mi3-3) | Cornell engineers need to know required environment variables for FactSet SQL access | [`b9f3d45`](https://github.com/kbsimms/cornell-semiconductor-project/commit/b9f3d45) | `.env.example` |
| Create `configs/README.md` (Mi7-4) | Canonical codebook version undocumented; config directory purposes unclear | [`b9f3d45`](https://github.com/kbsimms/cornell-semiconductor-project/commit/b9f3d45) | `configs/README.md` |
| Document CUDA determinism design (M5-1) | Replaced bare comment with explanation of why strict determinism is disabled | [`b9f3d45`](https://github.com/kbsimms/cornell-semiconductor-project/commit/b9f3d45) | `node2vec/run_node2vec_eval.py` |

---

## Unicode Cleanup — Emoji, Ligatures, and Zero-Width Characters

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Strip 157 emoji from DoD supply chain scripts and README | Emoji used as log/status indicators replaced with ASCII tags (`[OK]`, `[ERR]`, `[WARN]`, etc.) | [`062e3d7`](https://github.com/kbsimms/cornell-semiconductor-project/commit/062e3d7) | `dod_supply_chain_analysis/*.py`, `dod_supply_chain_analysis/README.md` |
| Strip emoji from TGNN and n2v_temporal model files; replace arrow in shipping README | Checkmark emoji in test scaffolds replaced with `[OK]`/`[PASS]`; U+279D arrow replaced with `->` | [`e184a5f`](https://github.com/kbsimms/cornell-semiconductor-project/commit/e184a5f) | `src/tgnn/model.py`, `src/n2v_temporal/model.py`, `src/data_processing/shipping/README.md` |
| Replace 87 Unicode ligatures and remove zero-width space in references.bib | PDF copy-paste artifacts (ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi) in abstract fields; 1 U+200B zero-width space removed | [`a7b2579`](https://github.com/kbsimms/cornell-semiconductor-project/commit/a7b2579) | `report/references.bib` |

---

## Client-Readiness Audit — Track 4: LaTeX Integrity

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Fix hardcoded "Appendix~A.2" cross-reference | Caption pointed to wrong appendix section; replaced with `\ref{app:a5-seeds}` | [`954bebb`](https://github.com/kbsimms/cornell-semiconductor-project/commit/954bebb) | `report/appendixA.tex` |
| Update bib count in PROJECT_MAP.md (185→192) | Count was stale after Round 2 bibliography additions | [`954bebb`](https://github.com/kbsimms/cornell-semiconductor-project/commit/954bebb) | `PROJECT_MAP.md` |
| Add Overleaf root requirement to SETUP.md | Overleaf project root must be repo root, not `report/` — undocumented | [`954bebb`](https://github.com/kbsimms/cornell-semiconductor-project/commit/954bebb) | `SETUP.md` |

---

## Repository Cleanup and Documentation

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Remove internal audit reports from tracked files | 7 audit reports in `docs/client-readiness-audit/` were internal QA artifacts, not client deliverables; preserved in git history | [`a381868`](https://github.com/kbsimms/cornell-semiconductor-project/commit/a381868) | `docs/client-readiness-audit/` (7 files) |
| Regenerate architecture graphs with pyan3 2.1.0 | Removed stale backup file and EvolveGCN references; stripped absolute paths; created `scripts/regenerate_architecture.sh` | [`e4f500a`](https://github.com/kbsimms/cornell-semiconductor-project/commit/e4f500a) | `docs/architecture/`, `scripts/`, `.gitignore` |
| Fix PROJECT_MAP.md file counts | Corrected 162→164 files, 54K→53K lines, 149→148 non-init; added `calls_heuristics` to architecture table | [`4a0bf5f`](https://github.com/kbsimms/cornell-semiconductor-project/commit/4a0bf5f) | `PROJECT_MAP.md` |
| Add Google Drive inventory snapshot | Full recursive tree of 660 GB delivery Drive folder (2,680 files mapped) for final delivery reference | [`7d5ec18`](https://github.com/kbsimms/cornell-semiconductor-project/commit/7d5ec18) | `docs/drive_inventory_2026-02-25.txt` |
| Rewrite SETUP.md for final delivery | Two-path onboarding (Drive download vs GitHub clone + selective sync); add .env setup; fix rclone instructions that could destroy git history; remove stale periodic sync section | [`99cfe94`](https://github.com/kbsimms/cornell-semiconductor-project/commit/99cfe94) | `SETUP.md` |

---

## Source Change Audit (February 26, 2026)

| Change | Rationale | Commit | File(s) |
|--------|-----------|--------|---------|
| Create source change audit | Complete record of every bibliography source added/removed across all 134 commits, with full justifications | [`84f5359`](https://github.com/kbsimms/cornell-semiconductor-project/commit/84f5359) | `SOURCE_CHANGE_AUDIT.md` |
| Improve audit with forensic justifications | Added detailed rationale for every removal: git diff evidence, original bib records, replacement sources, assessment of correctness | [`436009b`](https://github.com/kbsimms/cornell-semiconductor-project/commit/436009b) | `SOURCE_CHANGE_AUDIT.md` |
| Add Williams 2002 (correct source) and restore Renesas fire | Per external collaboration: (1) Added `williams_17kg_2002` — the correct source for energy/materials footprint claim, replacing a Zotero mixup with Williams 2003 (securitization theory). (2) Restored `reuters_japanese_2021` as its own supply chain disruption example in the continuity risks passage | [`75a9ba1`](https://github.com/kbsimms/cornell-semiconductor-project/commit/75a9ba1) | `references.bib`, `01_introduction.tex` |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total commits | 42 (excluding audit analysis reports) |
| Bibliography entries | 178 → 194 |
| New sources added | 16 |
| Sources deleted | 10 (all duplicates, orphans, or replacements) |
| Wrong-source citations fixed | 13 (including Williams 2003 → 2002 Zotero mixup) |
| Missing citations added | 12 |
| Dead URLs fixed | 4 |
| Metadata corrections | 40+ fields across 30+ entries |
| Factual corrections | 2 (earthquake magnitude, wrong-event citation) |
| Unicode cleanup | 177 emoji stripped, 87 ligatures decomposed, 1 zero-width space removed |
| Brace balance | Verified after every commit |
| Source audit | Complete audit document with external contractor collaboration |
