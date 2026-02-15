# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static documentation website for the [Tapkee](https://github.com/lisitsyn/tapkee) dimension reduction library. A Clojure app generates `index.html` via Hiccup templating; static resources (JS, CSS, markdown, data) are served alongside it.

## Build & Development Commands

```bash
make build          # Compile Clojure (lein compile)
make static-html    # Full build: compile → generate index.html → copy resources → render preview image
make local          # Build + serve at http://localhost:8000
make clean          # Remove static/, target/, node_modules/
```

The build runs `lein run` which prints the full HTML to stdout, captured into `static/index.html`. All files from `resources/public/` are copied into `static/`.

### Deployment (AWS)

```bash
make terraform-deploy   # Build + terraform apply + S3 sync + CloudFront invalidation
make terraform-plan     # Preview infrastructure changes
```

## Architecture

### HTML Generation (`src/tapkee/core.clj`)

Single Clojure file that generates the entire page. Key data structures:

- **`all-methods`** — 18 dimension reduction algorithms, each with shortname, longname, and a markdown file in `resources/public/md/`
- **`all-graphical-examples`** — 5 interactive D3.js visualizations (promoters, words, cbcl, mnist, faces)
- **`all-usage-examples`** — 3 code examples, each with C++, Python, and R source variants displayed in Bootstrap nav-tabs

The page is a single-page app: algorithm descriptions, graphical examples, and code examples all render inside Bootstrap 5 modals opened from navbar dropdowns.

### Content Loading (client-side)

`resources/public/js/loaders.js` handles all dynamic content:
- `loadMarkdownContent(selector, file)` — fetches `.markdown` files, parses with marked.js, applies highlight.js and MathJax
- `loadSourceCode(selector, file)` — fetches source code files, applies syntax highlighting
- `loadReadmeContent()` — loads `md/README.markdown` into the main page body

### Static Resources (`resources/public/`)

- `md/` — Algorithm documentation (markdown with LaTeX math) and `README.markdown` (main page content)
- `code/` — Usage examples: `.cpp`, `.py`, `.r` source files and `.md` descriptions
- `data/` — Pre-computed embedding JSON files for D3.js visualizations
- `js/` — D3.js visualization scripts (one per graphical example) + utility scripts
- `css/styles.css` — Custom styles (gradient header, code block styling, modal blur)
- `img/` — Favicons, face/MNIST images for visualization tooltips, generated preview image

### Preview Image Generation

`render_preview.js` uses Puppeteer to render `render_preview.html` (a standalone D3 visualization) and screenshot it to `img/tapkee-preview.png` (1200x630) for social media og:image tags.

### Infrastructure (`terraform/`)

AWS stack: S3 (static hosting) → CloudFront (CDN with security headers, HTTPS) → Route53 + ACM (custom domain `tapkee.lisitsyn.me`). Separate S3 buckets for CloudFront access logs.

## Key Patterns

- All external libraries load from CDN (Bootstrap 5.3.3, jQuery 3.7.1, D3 v3, highlight.js, marked.js, MathJax 3). CDN URLs are centralized in `cdn-urls` map.
- Modals use a shared `modal` function; content-specific wrappers (`method-modal`, `graphical-example-modal`, `usage-example-modal`) handle different content types.
- Usage example modals render language tabs (C++/Python/R) using Bootstrap nav-tabs with `map-indexed` to set the first tab as active.
- D3 visualizations all follow the same pattern: load JSON from `data/`, create 400x400 SVG, map coordinates, color by category, add Bootstrap tooltips on hover.
