# LaTeX Writing Preferences

## Compilation

- For English papers or notes in this project, prefer `pdflatex` by default.
- Use `xelatex` only when the document actually requires it, for example because of Chinese content, fonts, or other engine-specific dependencies.

## Displayed equations

- Avoid line breaks in the middle of displayed equations.
- In `equation` and `equation*`, keep a short standalone definition-style displayed equation on **one source line** when practical.
- In `equation` and `equation*`, do not split one short definition across multiple source lines just to isolate `=`, `\\quad`, commas, or short trailing terms.
- In `align`, `align*`, `aligned`, `split`, `gathered`, `cases`, and similar displayed-math environments, keep **each equation row on one source line** when practical.
- In those environments, place source line breaks immediately **after `\\`** rather than splitting one equation row across multiple source lines.
- For multi-line derivations, use `aligned`, `split`, `cases`, or a similar inner environment inside `equation` or `equation*` when appropriate for the surrounding manuscript style.

## Equation environments

- Prefer `\begin{equation}...\end{equation}` or `\begin{equation*}...\end{equation*}` for displayed equations.
- **Do not use `\[...\]` for displayed equations.** Treat it as disallowed, not merely discouraged. When encountered during local formatting edits, convert it to `equation*` unless the surrounding manuscript style requires numbered display math.
- Avoid `\(...\)` for displayed equations.
- Use only `$...$` for inline math.
- Avoid `$$...$$` and plain `\displaymath`.

## Labels and references

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
