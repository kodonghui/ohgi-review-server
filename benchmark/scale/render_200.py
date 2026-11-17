"""Build the immutable 605-article corpus and its 200-page PDF."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from pypdf import PdfReader

from benchmark.scale.render_book import BASE, EXTENSION, HERE, render


HISTORICAL_SHA = "6c1375683ca14d09a529a201e2e9674285e94319cb9ce2d2c36d2ff163ff3ea5"
SCENARIOS = HERE / "scenarios_200.txt"
FACETS = ("접수와 일반 조건", "예외와 적용 범위", "완료 확인")
ARTICLE = re.compile(r"^### 제(\d+)조 ", re.MULTILINE)


def bytes_155() -> bytes:
    base = BASE.read_bytes()
    separator = b"\n" if base.endswith(b"\n") else b"\n\n"
    extension = EXTENSION.read_text(encoding="utf-8").split("\n\n", 1)[1].encode("utf-8")
    original = base + separator + extension
    if hashlib.sha256(original).hexdigest() != HISTORICAL_SHA:
        raise ValueError("보존할 제1~155조의 원문 해시가 달라졌습니다.")
    return original


def assemble() -> bytes:
    chapters = re.split(r"(?m)(?=^## )", SCENARIOS.read_text(encoding="utf-8").strip())
    article_no = 156
    blocks = []
    titles = set()
    for chapter_no, chapter in enumerate(filter(None, chapters), start=20):
        heading, *rows = (line for line in chapter.splitlines() if line.strip())
        if not heading.startswith(f"## 제{chapter_no}장 ") or len(rows) != 15:
            raise ValueError(f"제{chapter_no}장의 제목과 시나리오 수를 확인하세요.")
        blocks.append(heading)
        for line in rows:
            title, *clauses = (field.strip() for field in line.split("|"))
            if title in titles or len(clauses) != len(FACETS) or any(not body for body in clauses):
                raise ValueError(f"신규 조문 데이터 형식 오류. {title}")
            titles.add(title)
            for facet, body in zip(FACETS, clauses, strict=True):
                blocks.append(f"### 제{article_no}조 ({title} · {facet})\n{body}")
                article_no += 1
    if article_no != 606:
        raise ValueError("신규 조문 450개가 필요합니다.")
    return bytes_155() + b"\n\n" + ("\n\n".join(blocks) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/ohgi-200p"))
    parser.add_argument("--fonts", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    args = parser.parse_args()
    data = assemble()
    text = data.decode("utf-8").replace("\r\n", "\n")
    if list(map(int, ARTICLE.findall(text))) != list(range(1, 606)):
        raise ValueError("전체 조문 번호가 연속이어야 합니다.")
    args.output.mkdir(parents=True, exist_ok=True)
    md_path = args.output / "regulations-200p.md"
    pdf_path = args.output / "regulations-200p.pdf"
    render(text, pdf_path, args.fonts, page_count=200, last_article=605)
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    pdf_articles = [int(n) for page in pages for n in re.findall(r"(?m)^제(\d+)조 \(", page)]
    if len(pages) != 200 or pdf_articles != list(range(1, 606)):
        raise ValueError(f"PDF 검증 실패. 페이지 {len(pages)}, 조문 제목 {len(pdf_articles)}")
    md_path.write_bytes(data)
    report = {
        "pages": len(pages), "articles": len(pdf_articles), "new_articles": len(pdf_articles) - 155,
        "original_155_sha256": HISTORICAL_SHA, "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "new_rules": "synthetic, written for this benchmark",
    }
    (args.output / "corpus.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
