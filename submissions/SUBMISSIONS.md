# Submission guide — per platform

One source of truth is `../PAPER.md`. Each platform copy is derived from it:

| Platform | File to use | Format | Notes |
|---|---|---|---|
| **arXiv** | `paper.tex` | LaTeX (compiles as preprint) | category `cs.LG` (cross-list `cs.CL`); pick a license (CC-BY 4.0 recommended). Upload `paper.tex` (+ any `.bbl`); compile on arXiv or Overleaf. |
| **ICBINB** (*I Can't Believe It's Not Better*, NeurIPS workshop) | `paper.tex` → their template | LaTeX, **~4 pages + refs/appendix**, **non-archival** | The caught non-replication + the fp16/ceiling artifacts are the *exact* remit ("negative/surprising results done right"). Drop the body into the workshop's NeurIPS-style `.sty`, **anonymize** (remove author), trim to 4pp. |
| **NeurIPS / ICLR interpretability or safety workshop** (e.g. MechInterp, SoLaR, Re-Align) | `paper.tex` → their template | LaTeX, usually **4–9 pages**, often non-archival, **anonymized** | Lead with §5.4 (cross-family eval-gaming). Use the venue's `.sty` (e.g. `neurips_2024.sty` / `iclr2025_conference.sty`). |
| **Alignment Forum / LessWrong** | `blog.md` | Markdown / rich-text, **no length limit** | Paste `blog.md` directly (the editor accepts markdown). No anonymization. Best home for the eval-gaming finding + the "how a behavioral eval lies to you" angle. |
| **Personal site / Substack** | `blog.md` or `PAPER.html` | HTML/MD | `../PAPER.html` is a standalone styled render. |
| **GitHub** | the whole repo | code + `PAPER.md` + `README.md` | See `../README.md`; repo is public, MIT-licensed code. |

## Retargeting `paper.tex`

`paper.tex` is a pandoc-generated standalone LaTeX preprint (compiles on Overleaf as-is). To submit
to a LaTeX venue:

1. Create an Overleaf project from the venue's official template (`.sty` + skeleton `main.tex`).
2. Copy the body of `paper.tex` (everything between `\begin{document}` and `\end{document}`, minus
   the pandoc `\maketitle`) into the template's body.
3. **Anonymize** if the venue is double-blind (remove author/affiliation; scrub the GitHub URL and
   `real1900` from the text — they appear in the abstract footnote and §Reproduction).
4. **Trim to the page limit** — the natural cut for a 4-page version is: keep §3 (one table), §4
   (one paragraph), §5.4 + the XSTest table, §6; move the rest to an appendix.
5. Convert the References list to the venue's `.bst` (BibTeX) — entries are in `PAPER.md` §References
   with verified arXiv IDs for the two concurrent works (Cox 2603.01437; Abdelnabi & Salem 2505.14617).

## Before you submit (any venue)

- [ ] **Verify the current CFP + deadline** on the venue's official page — workshop dates change yearly
      and are not encoded here.
- [ ] Double-check the two concurrent-work citations against the live arXiv pages (already verified
      2026-06; re-confirm titles/authors before camera-ready).
- [ ] Confirm author name/affiliation and ORCID.
- [ ] If anonymized: grep the source for `real1900`, the repo URL, and "M4 / Apple" → these can be
      lightly de-anonymizing; keep the laptop-scale claim, drop the username.
- [ ] Pick a license (paper: CC-BY 4.0; code: MIT, already in the repo).
