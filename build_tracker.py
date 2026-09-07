#!/usr/bin/env python3
"""Generate tracker.html from CHALLENGE.md. Run after editing the markdown."""
import html
import re
from pathlib import Path

SRC = Path(__file__).parent / "CHALLENGE.md"
OUT = Path(__file__).parent / "tracker.html"
YEAR = 2026
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Mei": 5, "Jun": 6,
          "Jul": 7, "Agu": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Des": 12}

CSS = """
:root{--bg:#faf9f7;--card:#fff;--ink:#1c1a17;--dim:#6b665f;
      --line:#e5e1db;--accent:#c2410c;--done:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--card:#1f1e1b;--ink:#eceae6;
      --dim:#9b958c;--line:#332f2a;--accent:#fb923c;--done:#4ade80}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 ui-sans-serif,-apple-system,system-ui,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--dim);font-size:13px;margin:0 0 20px;white-space:pre-line}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0 6px}
.bar>i{display:block;height:100%;background:var(--done);width:0;transition:width .3s}
.count{font-size:13px;color:var(--dim);margin-bottom:8px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);
   margin:34px 0 12px;font-weight:600}
.day{background:var(--card);border:1px solid var(--line);border-radius:12px;
     padding:14px 16px;margin-bottom:10px}
.day.today{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.day>header{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.day h3{font-size:15px;margin:0;font-weight:600}
.tag{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.now{font-size:11px;padding:2px 8px;border-radius:99px;background:var(--accent);color:#fff}
ul{list-style:none;margin:0;padding:0}
li{display:flex;gap:10px;align-items:flex-start;padding:4px 0}
li input{margin:3px 0 0;accent-color:var(--done);flex:none}
li label{cursor:pointer}
li:has(:checked) label{color:var(--dim);text-decoration:line-through}
li.note{display:list-item;list-style:none;color:var(--dim);font-size:13px;padding-left:0}
code{font-size:.9em;background:var(--line);padding:1px 5px;border-radius:4px}
footer{margin-top:40px;font-size:12px;color:var(--dim)}
"""

JS = """
const boxes=[...document.querySelectorAll('input')];
const tally=()=>{const n=boxes.filter(b=>b.checked).length;
  fill.style.width=(n/boxes.length*100)+'%';
  count.textContent=`${n}/${boxes.length} selesai (${Math.round(n/boxes.length*100)}%)`;};
boxes.forEach(b=>b.addEventListener('change',tally));tally();
const d=document.querySelector(`.day[data-date="${new Date().toLocaleDateString('sv')}"]`);
if(d){d.classList.add('today');
  d.querySelector('header').insertAdjacentHTML('beforeend','<span class="now">hari ini</span>');}
"""


def inline(text):
    """Markdown inline -> HTML: `code`, **bold**, *italic*."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", out)


def build(md):
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    # Intro = everything before the first "##", minus the how-to-use blockquote.
    intro = [l for l in lines[1:] if not l.startswith("#")]
    intro = intro[:next((i for i, l in enumerate(intro) if l.startswith("---")), len(intro))]
    intro = [l for l in intro if l.strip() and not l.startswith("> **Cara pakai")]

    body, open_list, open_day = [], False, False

    def close():
        nonlocal open_list, open_day
        if open_list:
            body.append("</ul>")
            open_list = False
        if open_day:
            body.append("</section>")
            open_day = False

    for line in lines:
        if line.startswith("### "):
            close()
            head = line[4:].strip()
            m = re.search(r"(\d{1,2}) (\w{3})", head)
            date = (f"{YEAR}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
                    if m and m.group(2) in MONTHS else "")
            body.append(f'<section class="day" data-date="{date}"><header>'
                        f"<h3>{inline(head)}</h3>")
            open_day = True
        elif line.startswith("## "):
            close()
            body.append(f"<h2>{inline(line[3:].strip())}</h2>")
        elif open_day and re.fullmatch(r"`[^`]+`", line.strip()):
            body.append(f'<span class="tag">{html.escape(line.strip(" `"))}</span>')
        elif line.startswith("- ["):
            if open_day and not open_list:
                body.append("</header><ul>")
                open_list = True
            elif not open_list:
                body.append("<ul>")
                open_list = True
            checked = " checked" if line[3].lower() == "x" else ""
            i = len([b for b in body if "<input" in b])
            body.append(f'<li><input type="checkbox" id="t{i}"{checked}>'
                        f'<label for="t{i}">{inline(line[6:].strip())}</label></li>')
        elif line.startswith("- "):
            if not open_list:
                body.append("<ul>")
                open_list = True
            body.append(f'<li class="note">{inline(line[2:].strip())}</li>')
    close()

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{"<br>".join(inline(l) for l in intro)}</p>
<div class="bar"><i id="fill"></i></div>
<div class="count" id="count"></div>
{chr(10).join(body)}
<footer>Generated from <code>CHALLENGE.md</code> by <code>build_tracker.py</code>.
Centang di browser tidak tersimpan — minta Claude Code edit markdown-nya, lalu regenerate.</footer>
</div>
<script>{JS}</script>
"""


def test():
    md = "# T\n\n## P\n\n### Hari 1 — Senin, 7 Sep\n`Tag`\n\n- [x] done\n- [ ] todo\n"
    out = build(md)
    assert 'data-date="2026-09-07"' in out, "date parse"
    assert out.count("<input") == 2, "task count"
    assert out.count(" checked>") == 1, "checked state"
    assert '<span class="tag">Tag</span>' in out, "tag"
    assert out.count("<section") == out.count("</section>") == 1, "unbalanced section"
    assert out.count("<ul>") == out.count("</ul>"), "unbalanced list"
    print("ok")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test()
    else:
        OUT.write_text(build(SRC.read_text()))
        print(f"{OUT.name}: {build(SRC.read_text()).count('<input')} tasks")
