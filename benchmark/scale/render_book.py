"""Render a 50-page A4 experiment copy and its exact Markdown input."""

import argparse
import os
import re
from itertools import batched
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "data" / "ohgi_regulations.md"
EXTENSION = HERE / "extension.md"
WIDTH, HEIGHT = A4
MARGIN = 64
INK = colors.HexColor("#1B2934")
MUTED = colors.HexColor("#465C6B")
ACCENT = colors.HexColor("#176B69")
RULE = colors.HexColor("#CFDFDE")


def paragraphs(text):
    return [line.strip().removeprefix("- ") for line in text.splitlines() if line.strip()]


def wrap(text, font, size, width):
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and pdfmetrics.stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def pages_of(source):
    title, *chapters = re.split(r"(?m)^## ", source)
    intro = paragraphs(title)[1:]
    pages = []
    for number, chapter in enumerate(chapters, start=1):
        heading, *raw_articles = re.split(r"(?m)^### ", chapter)
        articles = []
        for raw in raw_articles:
            article, body = raw.split("\n", 1)
            articles.append((article, paragraphs(body)))
        if number <= 9:
            groups = [articles]
        elif number == 19:
            groups = [articles[:3], articles[3:6], *batched(articles[6:], 2)]
        else:
            groups = list(batched(articles, 3))
        for page_in_chapter, group in enumerate(groups):
            pages.append((heading.strip(), page_in_chapter, group, intro if number == 1 else []))
            intro = []
    return pages


def render(source, path, fonts):
    pdfmetrics.registerFont(TTFont("Malgun", str(fonts / "malgun.ttf")))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(fonts / "malgunbd.ttf")))
    pages = pages_of(source)
    assert len(pages) == 50
    pdf = Canvas(str(path), pagesize=A4, pageCompression=1, invariant=1)
    pdf.setTitle("오지 레스토랑 운영 규정 · 50페이지 실험판")
    pdf.setAuthor("오지 레스토랑 실습")
    text_width = WIDTH - MARGIN * 2
    for number, (chapter, page_in_chapter, articles, intro) in enumerate(pages, start=1):
        if page_in_chapter == 0:
            pdf.bookmarkPage(chapter)
            pdf.addOutlineEntry(chapter, chapter)
        pdf.setFillColor(ACCENT)
        pdf.setFont("Malgun-Bold", 11)
        pdf.drawString(MARGIN, HEIGHT - 51, "오지 레스토랑 운영 규정")
        pdf.setFillColor(MUTED)
        pdf.setFont("Malgun", 10)
        pdf.drawRightString(WIDTH - MARGIN, HEIGHT - 51, "가상 확장판 · 검색 실험용")
        pdf.setStrokeColor(RULE)
        pdf.line(MARGIN, HEIGHT - 64, WIDTH - MARGIN, HEIGHT - 64)
        pdf.setFillColor(INK)
        pdf.setFont("Malgun-Bold", 23)
        pdf.drawString(MARGIN, HEIGHT - 111, chapter)

        y = HEIGHT - 156
        body_size = 10.8 if number <= 9 else 12.2
        leading = 15.8 if number <= 9 else 21.5
        for line in intro:
            pdf.setFont("Malgun", body_size)
            for wrapped in wrap(line, "Malgun", body_size, text_width):
                pdf.drawString(MARGIN, y, wrapped)
                y -= leading
        if intro:
            y -= 16
        for heading, body in articles:
            pdf.setFillColor(ACCENT)
            pdf.setFont("Malgun-Bold", 14)
            pdf.drawString(MARGIN, y, heading)
            y -= 22 if number <= 9 else 26
            pdf.setFillColor(INK)
            pdf.setFont("Malgun", body_size)
            for paragraph in body:
                for line in wrap(paragraph, "Malgun", body_size, text_width):
                    pdf.drawString(MARGIN, y, line)
                    y -= leading
                y -= 3
            y -= 12 if number <= 9 else 17
        if y < 64:
            raise ValueError(f"{number}페이지 본문이 아래 여백을 넘었습니다.")
        pdf.setStrokeColor(RULE)
        pdf.line(MARGIN, 57, WIDTH - MARGIN, 57)
        pdf.setFillColor(MUTED)
        pdf.setFont("Malgun", 9)
        pdf.drawString(MARGIN, 39, "1–35조 교재 본문 · 36–155조 가상 추가 조문")
        pdf.drawRightString(WIDTH - MARGIN, 39, f"{number} / {len(pages)}")
        pdf.showPage()
    pdf.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    args = parser.parse_args()
    # 원본 35조의 바이트를 유지한다.
    base = BASE.read_bytes()
    separator = b"\n" if base.endswith(b"\n") else b"\n\n"
    extension = EXTENSION.read_text(encoding="utf-8").split("\n\n", 1)[1]
    combined = base + separator + extension.encode("utf-8")
    source = combined.decode("utf-8").replace("\r\n", "\n")
    ids = [int(article) for article in re.findall(r"(?m)^### 제(\d+)조 ", source)]
    assert ids == list(range(1, 156))
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    render(source, args.pdf, args.fonts)
    args.markdown.write_bytes(combined)


if __name__ == "__main__":
    main()
