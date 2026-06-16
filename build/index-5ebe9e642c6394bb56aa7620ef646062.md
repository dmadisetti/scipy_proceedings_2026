---
title: Content-Addressed Caching for Reactive Notebooks
subtitle: A System for Cell-Level Cache Reuse in marimo
venue:
  title: SciPy 2026
  url: https://www.scipy2026.scipy.org/
authors:
  - name: Dylan Madisetti
    email: dylan@marimo.io
    orcid: 0000-0003-1269-4989
    corresponding: true
    affiliations:
      - marimo
  - name: Myles Scolnick
    email: myles@marimo.io
    affiliations:
      - marimo
  - name: Akshay Agrawal
    email: akshay@marimo.io
    affiliations:
      - marimo
license: CC-BY-4.0
open_access: true
downloads:
  - file: paper.pdf
    title: Paper (PDF)
  - file: ../main.md
    title: Notebook source (main.md)
---

+++ {"part": "abstract"}

:::{include} _abstract.md
:::

+++

## Read it your way

- **[The paper, live](./paper.md)** — the same notebook that builds the PDF, rendered with reactive marimo cells: drag the seed slider and watch the cache keys re-derive in your browser.
- **[Paper (PDF)](./paper.pdf)** — the typeset proceedings version.
- **[Demos](./demos/wasm-cache.md)** — the capabilities of the cache, interactive.
- **[Poster](./poster.md)** — presented at the SciPy 2026 poster session.

## The one-figure version

:::{figure} figs/fig1_dispatch.png
:width: 100%

The cache key dispatch (left) and the derivation over a full cell (right): the same parse that schedules a cell yields its cache key.
:::
