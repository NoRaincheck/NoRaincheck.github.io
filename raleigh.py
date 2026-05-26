"""Raleigh — minimal static site generator from markdown with front-matter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


def parse_front_matter(text: str) -> tuple[dict | None, str]:
    """Extract front-matter and body from markdown text.

    Front-matter is delimited by ``---`` at the top of the file.  Keys are
    ``str : value`` on single lines; values must be JSON-parsable.

    Returns ``(meta_dict, body_text)`` — meta is None if no front-matter found.
    """
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return None, text

    end = stripped.index("---", 3)
    fm_block = stripped[4:end].strip()
    body = stripped[end + 3 :].lstrip("\n")

    meta: dict = {}
    for line in fm_block.splitlines():
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        key = line[:colon_idx].strip().lower()
        val_raw = line[colon_idx + 1 :].strip()
        try:
            meta[key] = json.loads(val_raw)
        except (json.JSONDecodeError, ValueError):
            meta[key] = val_raw

    return meta or None, body


# ---------------------------------------------------------------------------
# Minimal markdown → HTML converter (no external deps)
# ---------------------------------------------------------------------------


def _inline(text: str) -> str:
    """Process inline markup."""
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_html(md_text: str) -> str:
    """Convert markdown text to HTML.

    Handles headings, bold/italic, code, links, images, lists, blockquotes,
    horizontal rules, and paragraphs.
    """
    lines = md_text.splitlines()
    html_parts: list[str] = []
    i = 0
    n = len(lines)

    def _flush(block_lines: list[str]) -> None:
        rendered = _render_block(block_lines)
        if rendered:
            html_parts.append(rendered)

    current_block: list[str] = []

    while i < n:
        line = lines[i]

        # Blank line ends a block
        if not line.strip():
            _flush(current_block)
            current_block = []
            i += 1
            continue

        # Code fence
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            _flush(current_block)
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                escaped = (
                    lines[i]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                code_lines.append(escaped)
                i += 1
            lang = m.group(1) or ""
            tag = "pre" if not lang else f'pre class="language-{lang}"'
            html_parts.append(f"<{tag}><code>{''.join(code_lines)}\n</code></{tag}>")
            current_block = []
            i += 1
            continue

        # Heading
        hm = re.match(r"^(#{1,6})\s+(.*)", line)
        if hm:
            _flush(current_block)
            level = len(hm.group(1))
            content = _inline(hm.group(2).strip())
            html_parts.append(f"<h{level}>{content}</h{level}>")
            current_block = []
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line):
            _flush(current_block)
            html_parts.append("<hr>")
            current_block = []
            i += 1
            continue

        # Table
        if "|" in line and re.match(r"^\|.*\|", line):
            _flush(current_block)
            table_lines: list[str] = [line]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                # Skip separator rows like |---|---| (only contain -, :, spaces, pipes)
                cells_raw = row_text.split("|")[1:-1]
                if all(re.match(r"^[\s:\-]+$", c) for c in cells_raw):
                    i += 1
                    continue
                table_lines.append(row_text)
                i += 1
            rows: list[list[str]] = []
            hdr_len = max(
                1, len([c for c in table_lines[0].split("|")[1:-1] if c.strip()])
            )
            for tl in table_lines:
                cells = [c.strip() for c in tl.split("|")[1:-1] if c.strip()]
                # Ensure trailing pipe gives correct count
                if tl.endswith("|") and len(cells) < hdr_len:
                    cells.append("")
                if not rows or len(cells) == hdr_len:
                    rows.append(cells)
            if rows:
                header_cells = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body_rows = "".join(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                    for row in rows[1:]
                )
                html_parts.append(
                    f"<table><thead><tr>{header_cells}</tr></thead><tbody>{body_rows}</tbody></table>"
                )
            continue

        # Blockquote
        if line.strip().startswith(">"):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                bq_text = re.sub(r"^\s*>\s?", "", lines[i])
                bq_lines.append(bq_text)
                i += 1
            _flush(current_block)
            inner = md_to_html("\n".join(bq_lines))
            html_parts.append(f"<blockquote>{inner}</blockquote>")
            continue

        # Unordered list
        if re.match(r"^\s*[-*+]\s+", line):
            _flush(current_block)
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                content = re.sub(r"^\s*[-*+]\s+", "", lines[i])
                items.append(f"<li>{_inline(content)}</li>")
                i += 1
            html_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        # Ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            _flush(current_block)
            items: list[str] = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                content = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                items.append(f"<li>{_inline(content)}</li>")
                i += 1
            html_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        current_block.append(line)
        i += 1

    _flush(current_block)
    return "\n".join(html_parts)


def _render_block(block: list[str]) -> str | None:
    """Render a single paragraph block."""
    text = " ".join(block)
    if not text.strip():
        return None
    return f"<p>{_inline(text)}</p>"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_date(val: str | datetime | date) -> date | None:
    """Parse YYYY-MM or YYYY-MM-DD into a date object."""
    if isinstance(val, (datetime, date)):
        return val.date() if hasattr(val, "date") else val
    if not isinstance(val, str):
        return None
    try:
        if len(val) == 7 and "-" in val:
            return datetime.strptime(val, "%Y-%m").date()
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Site generation
# ---------------------------------------------------------------------------

_CSS = """
/* Raleigh — Catppuccin Macchiato */
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 16px; scroll-behavior: smooth; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
    line-height: 1.7;
    color: #cad3f5;
    background: #24273a;
    margin: 0; padding: 0;
}
.site-header {
    background: #1e2030; color: #cad3f5;
    padding: 0.75rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
}
.site-header a { color: #cad3f5; text-decoration: none; font-weight: 600; }
nav.site-nav { display: flex; gap: 1rem; flex-wrap: wrap; }
nav.site-nav a {
    color: #a5adcb; text-decoration: none; font-size: 0.9rem;
}
nav.site-nav a:hover { color: #cad3f5; text-decoration: underline; }
.wrapper {
    max-width: 72em; margin: 0 auto; padding: 1.5rem;
    display: grid; grid-template-columns: 220px 1fr; gap: 2rem;
}
.sidebar { position: sticky; top: 4rem; align-self: start; }
.sidebar h3 {
    font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #939ab7; margin-bottom: 0.25rem;
}
.sidebar ul { list-style: none; padding: 0; margin: 0 0 1.5rem 0; }
.sidebar li a {
    display: block; padding: 0.3rem 0.6rem; text-decoration: none;
    color: #cad3f5; border-radius: 4px; font-size: 0.95rem;
}
.sidebar li a:hover { background: #363a4f; }
main { min-width: 0; }
main h1 { margin-top: 0; font-size: 2rem; color: #ed8796; }
main h2 { border-bottom: 1px solid #5b6078; padding-bottom: 0.3em; color: #f5a97f; }
main h3 { color: #eed49f; }
main h4 { color: #a6da95; }
main h5 { color: #7dc4e4; }
main h6 { color: #b7bdf8; }
article.post-entry { margin-bottom: 1.5rem; }
article.post-entry h2 { border: none; padding: 0; font-size: 1.3rem; color: #cad3f5; }
article.post-entry .post-date {
    color: #6e738d; font-size: 0.85rem; margin-bottom: 0.25rem;
}
article.post-entry .post-tags span {
    display: inline-block; background: #363a4f;
    padding: 0.1em 0.5em; border-radius: 3px;
    font-size: 0.75rem; margin-right: 0.25rem; color: #a5adcb;
}
pre {
    background: #1e2030; padding: 1em; overflow-x: auto;
    border-radius: 6px; font-size: 0.85rem; line-height: 1.5;
    color: #cad3f5;
}
code { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 0.85em; }
code, cmph { color: #ee99a0; }
pre code { color: inherit; }
em, i { color: #f5a97f; font-style: italic; }
strong, b { color: #ed8796; font-weight: 700; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #5b6078; padding: 0.5em 0.75em; text-align: left; }
th { background: #363a4f; font-weight: 600; color: #cad3f5; }
blockquote {
    margin: 1rem 0; padding: 0.5rem 1rem;
    border-left: 4px solid #6e738d; color: #b8c0e0;
}
.site-footer { text-align: center; padding: 2rem 1rem; font-size: 0.8rem; color: #6e738d; }
a { color: #8aadf4; text-decoration: none; } a:hover { color: #b7bdf8; text-decoration: underline; }
img { max-width: 100%; height: auto; border-radius: 4px; }
@media (max-width: 768px) {
    .wrapper { grid-template-columns: 1fr; }
    .sidebar { position: static; border-bottom: 1px solid #5b6078; padding-bottom: 1rem; margin-bottom: 1rem; }
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<header class="site-header">
    <a href="/">{site_title}</a>
    <nav class="site-nav">{nav_links}</nav>
</header>
<div class="wrapper">
<aside class="sidebar">
    <h3>Navigation</h3>
    <ul>{sidebar_items}</ul>
</aside>
<main>
{content}
</main>
</div>
<footer class="site-footer"><small>{footer}</small></footer>
</body>
</html>"""


def slugify(text: str) -> str:
    """Create a URL-friendly slug from text."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return s.strip("-").replace("--", "-") or "untitled"


class Site:
    """Minimal static site generator.

    Directory conventions (Jekyll / Hugo inspired):

        source/            root input directory
          index.md         homepage
          posts/           blog post collection
            2026-05-26-title.md
          about/
            index.md
          assets/          static files copied verbatim
    """

    def __init__(
        self,
        source_dir: str | Path = "source",
        output_dir: str | Path = "_site",
        site_title: str | None = None,
    ) -> None:
        self.source = Path(source_dir).resolve()
        self.output = Path(output_dir).resolve()
        self._config = self._load_config()
        self.site_title = site_title or self._config.get("site_title", "My Site")

    def _load_config(self) -> dict:
        config_path = self.source.parent / "config.json"
        if config_path.is_file():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}

    def _nav_links(self) -> str:
        links = self._config.get("nav", [{"name": "Home", "href": "/"}])
        return " ".join(f'<a href="{item["href"]}">{item["name"]}</a>' for item in links)

    def _sidebar_items(self) -> str:
        links = self._config.get("nav", [{"name": "Home", "href": "/"}])
        return "\n".join(f'<li><a href="{item["href"]}">{item["name"]}</a></li>' for item in links)

    def _format_page(self, title: str, content: str) -> str:
        css_escaped = _CSS.replace("</style>", "<\\/style>")
        return HTML_TEMPLATE.format(
            site_title=self.site_title,
            title=title,
            css=css_escaped,
            nav_links=self._nav_links(),
            sidebar_items=self._sidebar_items(),
            footer=self._config.get("footer", "Built with Raleigh"),
            content=content,
        )

    def build(self) -> int:
        """Generate the entire site. Returns number of pages written."""
        if not self.source.is_dir():
            print(
                f"Error: source directory {self.source} does not exist", file=sys.stderr
            )
            return 0

        # Collect all markdown files with front-matter
        posts: list[tuple[dict, Path]] = []
        pages: list[tuple[dict | None, str, Path]] = []

        for md_file in sorted(self.source.rglob("*.md")):
            raw = md_file.read_text(encoding="utf-8")
            meta, body = parse_front_matter(raw)
            rel = md_file.relative_to(self.source)

            if "date" in (meta or {}):
                posts.append((meta or {}, md_file))
            else:
                pages.append((meta, body, rel))

        # Sort posts by date descending
        posts.sort(
            key=lambda p: parse_date(p[0].get("date", "")) or date.min,
            reverse=True,
        )

        self.output.mkdir(parents=True, exist_ok=True)

        # Copy static assets
        assets = self.source / "assets"
        if assets.is_dir():
            for asset in assets.rglob("*"):
                dest = self.output / "assets" / asset.relative_to(assets)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if asset.is_file():
                    dest.write_bytes(asset.read_bytes())

        count = 0

        # Generate individual pages (no date in front-matter)
        for meta, body, rel in pages:
            html_body = md_to_html(body)
            title = (meta or {}).get("title", str(rel))
            out_path = self.output / rel.with_suffix(".html")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if slugify(title):
                out_path.write_text(
                    self._format_page(title, html_body), encoding="utf-8"
                )
                count += 1

        # Generate blog index page
        post_links = ""
        current_year: str | None = None
        for meta, _path in posts:
            d = parse_date(meta.get("date", ""))
            year_str = d.strftime("%Y") if d else "Unknown"
            if year_str != current_year:
                if current_year is not None:
                    post_links += "</ul>\n\n"
                post_links += f"<h2>{year_str}</h2>\n<ul>\n"
                current_year = year_str
            date_format = self._config.get("date_format", "%B %Y")
            date_str = d.strftime(date_format) if d else ""
            title = meta.get("title", "Untitled")
            tag_links = " ".join(
                f'<a href="/tags/{slugify(t)}.html">{t}</a>'
                for t in (meta.get("tags") or [])
            )
            link_path = f"posts/{slugify(title)}.html"
            post_links += (
                f'<article class="post-entry"><h2><a href="{link_path}">{title}</a></h2>'
                + (
                    f'\n<p class="post-date">{date_str}'
                    + (f"  · {tag_links}" if tag_links else "")
                    + "</p>"
                    if date_str or tag_links
                    else ""
                )
                + "\n</article>\n"
            )
        if current_year:
            post_links += "</ul>\n"

        blog_index = self._config.get("blog_index", "blog.html")
        blog_html = self._format_page(
            "Blog",
            "<h1>Blog Posts</h1>\n" + post_links,
        )
        (self.output / blog_index).write_text(blog_html, encoding="utf-8")
        count += 1

        # Generate index page listing posts
        home_post_list = ""
        for meta, _path in posts[:10]:
            d = parse_date(meta.get("date", ""))
            date_format = self._config.get("date_format", "%B %Y")
            date_str = d.strftime(date_format) if d else ""
            title = meta.get("title", "Untitled")
            tag_links = " ".join(
                f'<a href="/tags/{slugify(t)}.html">{t}</a>'
                for t in (meta.get("tags") or [])
            )
            link_path = f"posts/{slugify(title)}.html"
            home_post_list += (
                f'<article class="post-entry"><h2><a href="{link_path}">{title}</a></h2>'
                + (
                    f'\n<p class="post-date">{date_str}'
                    + (f"  · {tag_links}" if tag_links else "")
                    + "</p>"
                    if date_str or tag_links
                    else ""
                )
                + "\n</article>"
            )
        blog_index = self._config.get("blog_index", "blog.html")
        index_html = self._format_page(
            "Home",
            (home_post_list or "<p>No posts yet.</p>")
            + f'<p style="margin-top:2rem"><a href="{blog_index}">View all posts →</a></p>',
        )
        (self.output / "index.html").write_text(index_html, encoding="utf-8")
        count += 1

        # Generate per-page HTML for each post
        for meta, md_file in posts:
            body = parse_front_matter(md_file.read_text(encoding="utf-8"))[1]
            html_body = md_to_html(body)
            title = meta.get("title", "Untitled")
            d = parse_date(meta.get("date", ""))
            date_str = d.strftime(self._config.get("date_format_full", "%B %d, %Y")) if d else ""
            tags = meta.get("tags", []) or []
            tag_links = " ".join(
                f'<a href="/tags/{slugify(t)}.html">{t}</a>'
                for t in tags
            )

            post_html = self._format_page(
                title,
                f'<h1>{title}</h1>\n<p class="post-date">{date_str}'
                + (f"  · {tag_links}" if tag_links else "")
                + "</p>\n\n"
                + html_body,
            )
            out_path = self.output / "posts" / f"{slugify(title)}.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(post_html, encoding="utf-8")
            count += 1

        # Generate tag index pages
        tags: dict[str, list[tuple[dict, date]]] = {}
        for meta, _path in posts:
            for t in meta.get("tags") or []:
                d = parse_date(meta.get("date", "")) or date.min
                tags.setdefault(str(t), []).append((meta, d))

        tag_dir = self.output / "tags"
        tag_dir.mkdir(parents=True, exist_ok=True)
        for t, items in sorted(tags.items()):
            items.sort(key=lambda x: x[1], reverse=True)
            link_items = "".join(
                f'<article class="post-entry"><h2><a href="/posts/{slugify(str(m.get("title", "")))}.html">{m.get("title", "Untitled")}</a></h2></article>\n'
                for m, _d in items
            )
            tag_html = self._format_page(
                f"Posts tagged '{t}'",
                f"<h1>Posts tagged '{t}'</h1>\n{link_items}",
            )
            (tag_dir / f"{slugify(t)}.html").write_text(tag_html, encoding="utf-8")

        print(f"Built {count} pages → {self.output}")
        return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raleigh",
        description="Minimal static site generator from markdown with front-matter.",
    )
    parser.add_argument(
        "source", nargs="?", default="source", help="Source directory (default: source)"
    )
    parser.add_argument(
        "-o", "--output", default="_site", help="Output directory (default: _site)"
    )
    parser.add_argument("--title", help="Site title (default: config.json site_title, or 'My Site')")

    args = parser.parse_args(argv)
    site = Site(source_dir=args.source, output_dir=args.output, site_title=args.title)
    return 0 if site.build() > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
