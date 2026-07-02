"""Build a presentation-ready HTML report from a Jupyter notebook."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

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


def enhance_html_report(
    html_path: Path,
    *,
    title: str,
    output_path: Path | None = None,
    skip_prefixes: tuple[str, ...] = DEFAULT_SKIP_SECTION_PREFIXES,
) -> Path:
    """Add sidebar navigation and presentation styling to an nbconvert HTML file."""
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    if soup.head is None:
        raise ValueError(f"Invalid HTML file: {html_path}")

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
    """Run nbconvert to produce a raw HTML export without code cells."""
    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "html",
        "--no-input",
        "--no-prompt",
        "--embed-images",
        str(notebook_path),
    ]
    subprocess.run(command, check=True, cwd=notebook_path.parent)
    return notebook_path.with_suffix(".html")


def export_notebook_report(
    notebook_path: str | Path,
    *,
    title: str | None = None,
    output_path: str | Path | None = None,
    skip_prefixes: tuple[str, ...] = DEFAULT_SKIP_SECTION_PREFIXES,
) -> Path:
    """Export a notebook to a presentation-ready HTML report."""
    notebook = Path(notebook_path).resolve()
    if not notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    report_title = title or notebook.stem.replace("_", " ")
    raw_html = export_notebook_to_html(notebook)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else notebook.with_name(f"{notebook.stem}_report.html")
    )
    return enhance_html_report(
        raw_html,
        title=report_title,
        output_path=destination,
        skip_prefixes=skip_prefixes,
    )


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
