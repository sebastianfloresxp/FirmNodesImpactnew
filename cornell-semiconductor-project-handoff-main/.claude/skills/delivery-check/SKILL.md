---
name: delivery-check
description: Pre-delivery audit — scan for PII, internal-only content, and files not fit for client or production view, then check git tracking against the TODO.md remediation tiers
---

You are performing a pre-delivery audit of this repository. Work through each area systematically:

**1. PII and personal identifiers**
- Search tracked files for personal names, email addresses, GitHub usernames, and local machine paths (e.g. `/home/<username>/`).
- Flag any that belong to developers and should not appear in a client-facing repo.

**2. Internal-only content not fit for client view**
- Look for contract details (PO numbers, dollar amounts, milestone payment language, SOW references).
- Look for internal QA process artifacts, delivery readiness scores, audit framing, and contractor-internal notes.
- Look for references to internal tooling, personal development machine configurations, or unpublished model codenames.

**3. Credentials and secrets**
- Search for API keys, tokens, passwords, and PAT references in tracked files.
- Confirm `.env` is gitignored and not tracked.

**4. Git tracking — Tier 2 (files that must not be in HEAD)**
- Run `git ls-files` against each path listed under Tier 2 in TODO.md.
- For any still-tracked path, note that `git rm --cached` is needed and the path should be in `.gitignore`.

**5. Git tracking — Tier 3 (content to sanitize in tracked files)**
- For each item listed under Tier 3 in TODO.md, open the referenced file and line, and confirm the content has been updated or removed.

**6. History rewrite readiness — Tier 1**
- Confirm Tier 2 and Tier 3 are fully resolved before recommending `git filter-repo`.
- Remind the user that this rewrites all commit SHAs and requires coordinated force-push across all clones.

Report findings as a checklist: checked done, x outstanding, warning needs review.
