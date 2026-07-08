"""Build a presentation-ready HTML report from a Jupyter notebook."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

DEFAULT_SKIP_SECTION_PREFIXES = ("imports", "diretório", "ordem")

PRESENTATION_CSS = """
[data-mime-type='application/vnd.jupyter.stderr'] { display: none !important; }
.jp-Cell.jp-mod-noOutputs.jp-mod-noInput { display: none !important; }
.anchor-link { display: none !important; }

:root {
  --sidebar-bg: #0f172a;
  --sidebar-text: #e2e8f0;
  --sidebar-muted: #94a3b8;
  --sidebar-hover: #1e293b;
  --sidebar-active: #334155;
  --content-bg: #ffffff;
  --content-text: #0f172a;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--content-bg);
  color: var(--content-text);
}

#sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: 280px;
  height: 100vh;
  overflow-y: auto;
  background: var(--sidebar-bg);
  color: var(--sidebar-text);
  padding: 24px 16px;
  box-sizing: border-box;
  z-index: 1000;
}

#sidebar h2 {
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--sidebar-muted);
  margin: 0 0 16px;
  line-height: 1.4;
}

#sidebar a {
  display: block;
  color: var(--sidebar-text);
  text-decoration: none;
  padding: 8px 10px;
  border-radius: 8px;
  margin-bottom: 4px;
  font-size: 14px;
  line-height: 1.35;
}

#sidebar a:hover,
#sidebar a.active {
  background: var(--sidebar-hover);
}

#sidebar a.active {
  background: var(--sidebar-active);
  font-weight: 600;
}

#sidebar a.level-2 {
  padding-left: 22px;
  font-size: 13px;
  color: #cbd5e1;
}

#main {
  margin-left: 300px;
  padding: 32px 40px 80px;
  max-width: 1200px;
}

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2 {
  scroll-margin-top: 24px;
}

.jp-OutputArea img,
.jp-RenderedImage img {
  max-width: 100%;
  height: auto;
}

.dataframe {
  font-size: 13px;
}

#report-footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid #e2e8f0;
  color: #64748b;
  font-size: 12px;
}

@media print {
  #sidebar {
    display: none;
  }

  #main {
    margin-left: 0;
    padding: 0;
    max-width: none;
  }
}

@media (max-width: 960px) {
  #sidebar {
    position: static;
    width: 100%;
    height: auto;
  }

  #main {
    margin-left: 0;
    padding: 24px 16px 48px;
  }
}
"""

SIDEBAR_SCRIPT = """
const links = [...document.querySelectorAll('#sidebar a')];
const sections = links
  .map((anchor) => document.querySelector(anchor.getAttribute('href')))
  .filter(Boolean);

function updateActiveLink() {
  if (!sections.length) {
    return;
  }

  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= 120) {
      current = section;
    }
  }

  links.forEach((anchor) => anchor.classList.remove('active'));
  const active = links.find(
    (anchor) => anchor.getAttribute('href') === `#${current.id}`
  );
  if (active) {
    active.classList.add('active');
  }
}

window.addEventListener('scroll', updateActiveLink, { passive: true });
updateActiveLink();
"""


def _normalize_heading_text(text: str) -> str:
    return text.replace("¶", "").strip()


def _should_skip_section(
    section_id: str,
    heading_text: str,
    skip_prefixes: tuple[str, ...],
) -> bool:
    normalized_id = unquote(section_id).strip().lower()
    normalized_text = heading_text.strip().lower()
    return any(
        normalized_id.startswith(prefix) or normalized_text.startswith(prefix)
        for prefix in skip_prefixes
    )


def _build_sidebar_items(
    soup: BeautifulSoup,
    skip_prefixes: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for heading in soup.select("h1, h2"):
        section_id = heading.get("id")
        if not section_id:
            continue

        text = _normalize_heading_text(heading.get_text())
        if not text or _should_skip_section(section_id, text, skip_prefixes):
            continue

        level = "1" if heading.name == "h1" else "2"
        items.append((section_id, text, level))
    return items


def _get_active_notebook_path() -> Path | None:
    """Return the active notebook path when running inside VS Code, Cursor, or Jupyter."""
    candidate_paths: list[Path] = []

    for namespace in (_get_ipython_namespace(), globals()):
        if namespace is None:
            continue
        for key in ("__vsc_ipynb_file__", "__session__"):
            raw_path = namespace.get(key)
            if raw_path:
                candidate_paths.append(Path(str(raw_path)).expanduser())

    for candidate in candidate_paths:
        if candidate.suffix == ".ipynb" and candidate.exists():
            return candidate.resolve()

    return None


def _get_ipython_namespace() -> dict | None:
    try:
        from IPython import get_ipython
    except ImportError:
        return None

    ipython = get_ipython()
    if ipython is None:
        return None

    return ipython.user_ns


def _resolve_notebook_path(notebook_path: str | Path | None) -> Path:
    """Resolve a notebook path, preferring the active notebook in VS Code/Cursor."""
    active = _get_active_notebook_path()

    if notebook_path is None:
        if active is None:
            raise ValueError(
                "Notebook path is required outside of VS Code, Cursor, or Jupyter."
            )
        return active

    candidate = Path(notebook_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    search_roots: list[Path] = []
    if active is not None:
        search_roots.append(active.parent)
    search_roots.append(Path.cwd())

    seen_roots: set[Path] = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        resolved = (resolved_root / candidate).resolve()
        if resolved.exists():
            return resolved

    if active is not None and active.name == candidate.name:
        return active.resolve()

    resolved = (Path.cwd() / candidate).resolve()
    if resolved.exists():
        return resolved

    raise FileNotFoundError(f"Notebook not found: {notebook_path}")


def _prepare_notebook_for_export(notebook_path: Path) -> None:
    """Wait for editor autosave and warn when the on-disk notebook looks stale."""
    active = _get_active_notebook_path()
    if active is None or active.resolve() != notebook_path.resolve():
        return

    initial_mtime = notebook_path.stat().st_mtime
    deadline = time.monotonic() + 2.0
    latest_mtime = initial_mtime

    while time.monotonic() < deadline:
        time.sleep(0.15)
        latest_mtime = notebook_path.stat().st_mtime
        if latest_mtime > initial_mtime:
            initial_mtime = latest_mtime
            deadline = time.monotonic() + 0.75

    age_seconds = time.time() - latest_mtime
    if age_seconds > 120:
        saved_at = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(
            "WARNING: Notebook file has not been saved recently "
            f"(last saved at {saved_at}). "
            "Save the notebook (Cmd+S / Ctrl+S) before exporting to include the latest outputs.",
            file=sys.stderr,
        )


def enhance_html_report(
    html_path: Path,
    *,
    title: str,
    output_path: Path | None = None,
    skip_prefixes: tuple[str, ...] = DEFAULT_SKIP_SECTION_PREFIXES,
    generated_at: datetime | None = None,
    source_notebook: Path | None = None,
) -> Path:
    """Add sidebar navigation and presentation styling to an nbconvert HTML file."""
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    if soup.head is None:
        raise ValueError(f"Invalid HTML file: {html_path}")

    for attrs in (
        {"http-equiv": "Cache-Control", "content": "no-cache, no-store, must-revalidate"},
        {"http-equiv": "Pragma", "content": "no-cache"},
        {"http-equiv": "Expires", "content": "0"},
    ):
        meta_tag = soup.new_tag("meta")
        for key, value in attrs.items():
            meta_tag[key] = value
        soup.head.append(meta_tag)

    style_tag = soup.new_tag("style")
    style_tag.string = PRESENTATION_CSS
    soup.head.append(style_tag)

    nav_items = _build_sidebar_items(soup, skip_prefixes)
    sidebar = soup.new_tag("nav", id="sidebar")
    sidebar_title = soup.new_tag("h2")
    sidebar_title.string = title
    sidebar.append(sidebar_title)

    for section_id, text, level in nav_items:
        anchor = soup.new_tag("a", href=f"#{section_id}")
        anchor["class"] = [f"level-{level}"]
        anchor.string = text
        sidebar.append(anchor)

    if soup.body is None:
        raise ValueError(f"Invalid HTML file: {html_path}")

    main_wrapper = soup.new_tag("div", id="main")
    for child in list(soup.body.contents):
        main_wrapper.append(child.extract())

    generated_at = generated_at or datetime.now(timezone.utc)
    footer = soup.new_tag("footer", id="report-footer")
    footer_parts = [
        f"Report generated at {generated_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]
    if source_notebook is not None:
        footer_parts.append(f"Source notebook: {source_notebook.name}")
    footer.string = " | ".join(footer_parts)
    main_wrapper.append(footer)

    soup.body.clear()
    soup.body.append(sidebar)
    soup.body.append(main_wrapper)

    script_tag = soup.new_tag("script")
    script_tag.string = SIDEBAR_SCRIPT
    soup.body.append(script_tag)

    if soup.title:
        soup.title.string = title

    destination = output_path or html_path.with_name(f"{html_path.stem}_report.html")
    destination.write_text(str(soup), encoding="utf-8")
    return destination


def export_notebook_to_html(notebook_path: Path) -> Path:
    """Convert a notebook to raw HTML without code cells."""
    notebook = nbformat.read(notebook_path, as_version=4)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.embed_images = True

    body, _ = exporter.from_notebook_node(notebook)

    fd, temp_path = tempfile.mkstemp(
        suffix=".html",
        prefix=f"{notebook_path.stem}_",
        dir=notebook_path.parent,
    )
    os.close(fd)

    raw_html = Path(temp_path)
    raw_html.write_text(body, encoding="utf-8")
    return raw_html


def export_notebook_report(
    notebook_path: str | Path | None = None,
    *,
    title: str | None = None,
    output_path: str | Path | None = None,
    skip_prefixes: tuple[str, ...] = DEFAULT_SKIP_SECTION_PREFIXES,
) -> Path:
    """Export a notebook to a presentation-ready HTML report."""
    notebook = _resolve_notebook_path(notebook_path)
    _prepare_notebook_for_export(notebook)

    report_title = title or notebook.stem.replace("_", " ")
    generated_at = datetime.now(timezone.utc)
    raw_html = export_notebook_to_html(notebook)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else notebook.with_name(f"{notebook.stem}_report.html")
    )
    try:
        return enhance_html_report(
            raw_html,
            title=report_title,
            output_path=destination,
            skip_prefixes=skip_prefixes,
            generated_at=generated_at,
            source_notebook=notebook,
        )
    finally:
        raw_html.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Jupyter notebook to a presentation-ready HTML report."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Sidebar title for the report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <notebook>_report.html)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = export_notebook_report(
        args.notebook,
        title=args.title,
        output_path=args.output,
    )
    print(f"Report generated: {output}")


if __name__ == "__main__":
    main()
