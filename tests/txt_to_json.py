from pathlib import Path
import json
import re


INPUT_DIR = Path("converted_txt")
OUTPUT_DIR = Path("src") / "parsed"
OUTPUT_FILE = OUTPUT_DIR / "documents.jsonl"


def clean_line(line: str) -> str:
    """
    TXT 한 줄 정리
    """
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def read_text_file(path: Path) -> str:
    """
    UTF-8 우선으로 읽고, 실패하면 CP949도 시도
    """
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    # 그래도 실패하면 깨지는 문자는 무시하고 읽기
    return path.read_text(encoding="utf-8", errors="ignore")


def split_by_article(text: str) -> list[dict]:
    """
    제1조(목적), 제2조(정의) 같은 조항 단위로 분리.
    조항이 없으면 전체를 하나로 반환.
    """

    # 줄 정리
    lines = []
    for line in text.splitlines():
        line = clean_line(line)
        if line:
            lines.append(line)

    text = "\n".join(lines)

    # 제1조, 제1조의2, 제59조(성적평가) 패턴
    article_pattern = re.compile(
        r"(제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\([^)]+\))?)"
    )

    matches = list(article_pattern.finditer(text))

    if not matches:
        return [{
            "article": "",
            "title": "",
            "content": text
        }]

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        article_header = match.group(1).strip()
        content = text[start:end].strip()

        # 조항 번호 추출: 제59조, 제59조의2
        article_no_match = re.search(r"제\s*\d+\s*조(?:의\s*\d+)?", article_header)
        article_no = article_no_match.group(0).replace(" ", "") if article_no_match else ""

        # 제목 추출: (성적평가)
        title_match = re.search(r"\(([^)]+)\)", article_header)
        title = title_match.group(1).strip() if title_match else ""

        if len(content) >= 20:
            chunks.append({
                "article": article_no,
                "title": title,
                "content": content
            })

    return chunks


def make_doc_id(category: str, file_stem: str, idx: int) -> str:
    safe_category = re.sub(r"[^가-힣a-zA-Z0-9_]+", "_", category)
    safe_file = re.sub(r"[^가-힣a-zA-Z0-9_]+", "_", file_stem)
    return f"{safe_category}_{safe_file}_{idx:04d}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    txt_files = list(INPUT_DIR.rglob("*.txt"))
    print(f"총 TXT 파일 수: {len(txt_files)}")

    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for file_idx, txt_path in enumerate(txt_files, start=1):
            relative_path = txt_path.relative_to(INPUT_DIR)

            # converted_txt/학칙/파일.txt 라면 category = 학칙
            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "기타"

            source_file = txt_path.name
            file_stem = txt_path.stem

            text = read_text_file(txt_path)

            if not text.strip():
                print(f"[빈 파일] {relative_path}")
                continue

            chunks = split_by_article(text)

            print(f"[{file_idx}/{len(txt_files)}] {relative_path} → {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks, start=1):
                doc = {
                    "id": make_doc_id(category, file_stem, chunk_idx),
                    "category": category,
                    "source_file": source_file,
                    "article": chunk["article"],
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "path": str(txt_path).replace("\\", "/")
                }

                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("\n파싱 완료")
    print(f"총 chunk 수: {total_chunks}")
    print(f"저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()