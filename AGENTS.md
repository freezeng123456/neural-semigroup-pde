# Project Context

This project is a LaTeX note-writing workspace for mathematical notes related to `DC-PINNs_LaTeX_arxiv_2604.13723`.

## Repository Role

Unless the user explicitly asks otherwise, treat this directory primarily as a **note-writing / TeX-formatting workspace** for:

- standalone LaTeX notes,
- mathematical writeups,
- proposal-style notes,
- and note sections that may later be reused in a paper.

The local note-writing rules are recorded in:

- `latex-writing-preferences.md`
- `latex-note-formatter.agent.md`

## Main Goal

When working in this project, act as a specialist for **writing and revising LaTeX notes with clean mathematical formatting**.

Your job is to modify or create note-style `.tex` files so that they satisfy the project-local writing conventions while preserving:

- mathematical meaning,
- notation,
- macro definitions,
- labels and numbering,
- reference consistency,
- and authorial intent.

## Scope Discipline

- Do **not** invent assumptions, lemmas, propositions, theorem statements, proof steps, citations, or mathematical claims unless the user explicitly asks for such content creation.
- Do **not** broaden a local note-writing request into a whole-project rewrite unless the user explicitly asks for a broader pass.
- Do **not** silently change notation, macro definitions, numbering, or reference style beyond what is needed for the requested local edit.
- Prefer the **smallest reviewable edit** that enforces the local note-writing rules while keeping the mathematical exposition readable.

## Required Note-Writing Rules

### Compilation

- For English papers or notes in this project, prefer `pdflatex` by default.
- Use `xelatex` only when the document actually requires it, for example because of Chinese content, fonts, or other engine-specific dependencies.

### Displayed equations

- Avoid line breaks in the middle of displayed equations.
- In `equation` and `equation*`, keep a short standalone definition-style displayed equation on **one source line** when practical.
- In `equation` and `equation*`, do not split one short definition across multiple source lines just to isolate `=`, `\quad`, commas, or short trailing terms.
- In `align`, `align*`, `aligned`, `split`, `gathered`, `cases`, and similar displayed-math environments, keep **each equation row on one source line** when practical.
- In those environments, place source line breaks immediately **after `\\`** rather than splitting one equation row across multiple source lines.
- For multi-line derivations, use `aligned`, `split`, `cases`, or a similar inner environment inside `equation` or `equation*` when appropriate for the surrounding manuscript style.

### Equation environments

- Prefer `\begin{equation}...\end{equation}` or `\begin{equation*}...\end{equation*}` for displayed equations.
- **Do not use `\[...\]` for displayed equations.** Treat it as disallowed, not merely discouraged. When encountered during local formatting edits, convert it to `equation*` unless the surrounding manuscript style requires numbered display math.
- Avoid `\(...\)` for displayed equations.
- Use only `$...$` for inline math.
- Avoid `$$...$$` and plain `\displaymath`.

### Labels and references

- Replace vague phrases such as "the previous theorem" or "the above equation" with explicit LaTeX cross-references.
- Prefer a **uniform `\cref` / `\Cref` strategy** for theorem-like environments and numbered structural objects whenever the manuscript supports it.
- When editing existing references in this project, prefer `\Cref{...}` at the beginning of a sentence and `\cref{...}` mid-sentence.
- Prefer `\eqref{...}` for equations unless the surrounding local style clearly uses `\cref{...}` for equations as well; keep equation reference style locally consistent within the edited region.
- If the local manuscript already consistently uses `\cref` or `\Cref`, preserve and strengthen that local reference style rather than falling back to manual forms such as `Theorem~\ref{...}`.
- Use manual forms such as `Theorem~\ref{...}`, `Lemma~\ref{...}`, and `Proposition~\ref{...}` only when `cleveref`-style references are unavailable or clearly inconsistent with the local manuscript setup.
- If an explicit cross-reference is needed but the relevant object has no label, add a **local** `\label{...}` and update the corresponding references in the same edit.
- For `\begin{equation}...\end{equation}`, place `\label{...}` immediately after `\begin{equation}`.
- For `\begin{align}...\end{align}`, place `\label{...}` at the end of the relevant equation line.
- Do not rename labels unless the user asks, or the current edit truly requires adding a missing label.

## Working Method

When a user asks for note writing or note revision in this project, follow this workflow:

1. Start from the concrete anchor: the target file, section, paragraph, theorem, or equation block.
2. Read only the nearby context needed to preserve notation, labels, numbering, and local reference style.
3. Apply the formatting rules locally, keeping edits narrow and reviewable.
4. When a vague textual reference points to a nearby theorem, lemma, proposition, or equation, replace it with an explicit cross-reference in the local style.
5. When displayed math needs restructuring, prefer the smallest safe environment conversion that preserves numbering and references.
6. Keep a short standalone `equation` or `equation*` definition on one source line unless line length genuinely makes that impractical.

## Validation Guidance

- If an edit affects labels, cross-references, numbering, or equation environments, run a focused LaTeX validation step when feasible.
- Otherwise, avoid unnecessary heavy validation unless the user asks for it.
- Do not claim compilation success unless it was actually verified.

## Reporting Style

When reporting completed note-writing edits, briefly summarize:

- the local writing or formatting objective,
- any label/reference/numbering risk,
- which TeX structures were changed,
- and whether validation was run.
