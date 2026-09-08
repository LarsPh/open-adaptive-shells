"""Render README.md as a local document preview."""

import argparse
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme", type=Path, default=Path(__file__).resolve().parents[1] / "README.md"
    )
    readme = parser.parse_args().readme.resolve()
    root = readme.parent
    markdown = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = markdown.parse(readme.read_text(encoding="utf-8"))
    for token in tokens:
        if token.type != "inline" or not token.children:
            continue
        links = [child for child in token.children if child.type == "link_open"]
        if len(links) == 1 and links[0].attrGet("href").endswith(".mp4"):
            href = escape(links[0].attrGet("href"), quote=True)
            description = markdown.renderer.renderInline(token.children, markdown.options, {})
            token.children = []
            token.type = "html_inline"
            token.content = (
                f'{description}<video controls playsinline preload="metadata" src="{href}"></video>'
            )
    body = markdown.renderer.render(tokens, markdown.options, {})
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<base href="../../"><title>Adaptive Shells - README</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#1f2328;
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1012px;margin:32px auto;padding:32px 48px;border:1px solid #d1d9e0}
h1,h2{line-height:1.25;border-bottom:1px solid #d1d9e0;padding-bottom:.3em}
h1{font-size:32px}h2{font-size:24px;margin-top:32px}h3{font-size:20px}
a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}
img,video{display:block;max-width:100%;height:auto;margin:16px 0}
video{width:100%;background:#111}pre{padding:16px;background:#f6f8fa;overflow:auto;
font-size:13px;border-radius:6px}code{font-family:Consolas,monospace}
:not(pre)>code{font-size:85%;padding:.2em .4em;background:#eff1f3;border-radius:4px}
table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:6px 13px;
border:1px solid #d1d9e0}tr:nth-child(even){background:#f6f8fa}
@media(max-width:700px){main{margin:0;border:0;padding:20px}h1{font-size:28px}}
</style></head><body><main>""" + body + "</main></body></html>"
    destination = root / "outputs/presentation/README.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
