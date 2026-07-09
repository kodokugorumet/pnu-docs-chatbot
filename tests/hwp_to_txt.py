from pathlib import Path
import zlib
import struct
import olefile
import re
import unicodedata

DATA_DIR = Path("src") / "data" / "부산대학교" / "학칙"
OUT_DIR = Path("src") / "converted_txt" / "부산대학교" / "학칙"


def is_compressed_hwp(ole: olefile.OleFileIO) -> bool:
    """
    HWP FileHeader의 압축 여부 확인.
    bit 0이 1이면 BodyText가 zlib raw deflate로 압축되어 있음.
    """
    if not ole.exists("FileHeader"):
        return False

    header = ole.openstream("FileHeader").read()

    # HWP FileHeader에서 flags는 보통 offset 36~39에 위치
    flags = struct.unpack("<I", header[36:40])[0]
    return bool(flags & 0x01)


def get_section_names(ole: olefile.OleFileIO) -> list[str]:
    """
    BodyText/Section0, BodyText/Section1 ... 목록 수집
    """
    section_names = []

    for item in ole.listdir():
        # item 예: ['BodyText', 'Section0']
        if len(item) == 2 and item[0] == "BodyText" and item[1].startswith("Section"):
            section_names.append("/".join(item))

    # Section0, Section1 순서대로 정렬
    section_names.sort(key=lambda x: int(x.split("Section")[-1]))

    return section_names


def extract_text_from_section(data: bytes) -> str:
    """
    HWP section record에서 텍스트 레코드만 추출.
    레코드 헤더 4바이트:
    - 하위 10비트: tag id
    - 다음 10비트: level
    - 다음 12비트: size
    tag id 67이 문단 텍스트(HWPTAG_PARA_TEXT)
    """
    text_parts = []
    pos = 0
    size = len(data)

    while pos + 4 <= size:
        header = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        rec_size = (header >> 20) & 0xFFF

        # rec_size가 0xFFF면 실제 크기 4바이트 추가
        if rec_size == 0xFFF:
            if pos + 4 > size:
                break
            rec_size = struct.unpack_from("<I", data, pos)[0]
            pos += 4

        rec_data = data[pos:pos + rec_size]
        pos += rec_size

        # 67 = HWPTAG_PARA_TEXT
        if tag_id == 67:
            try:
                text = rec_data.decode("utf-16le", errors="ignore")
                text_parts.append(text)
            except Exception:
                pass

    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """
    HWP 추출 텍스트 후처리:
    - 제어문자 제거
    - 깨진 한자/중국어 영역 제거
    - RAG에 필요한 문자만 보존
    - 공백 정리
    """

    # 1. 제어문자 제거
    text = ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )

    # 2. 한자/중국어 영역 제거
    # 실제 한자 원문까지 제거될 수 있지만, 현재 깨진 문자 제거 목적상 적용
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)

    # 3. 허용 문자만 남기기
    # 한글, 영어, 숫자, 공백, 주요 문장부호, 조항 기호 유지
    text = re.sub(
        r'[^가-힣a-zA-Z0-9\s\.\,\[\]\(\)\:\-\~\%\/\+\=「」『』〈〉《》·ㆍ㎡℃°①-⑳Ⅰ-Ⅹ]',
        '',
        text
    )

    # 4. 탭을 공백으로
    text = re.sub(r'\t+', ' ', text)

    # 5. 한 줄 안의 연속 공백 정리
    text = re.sub(r'[ ]{2,}', ' ', text)

    # 6. 같은 문자가 과도하게 반복되는 경우 축소
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 7. 빈 줄 정리
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def convert_one_hwp(hwp_path: Path, output_path: Path) -> bool:
    try:
        ole = olefile.OleFileIO(str(hwp_path))
    except Exception as e:
        print(f"[열기 실패] {hwp_path} / {e}")
        return False

    try:
        compressed = is_compressed_hwp(ole)
        section_names = get_section_names(ole)

        if not section_names:
            print(f"[본문 없음] {hwp_path}")
            return False

        full_text_parts = []

        for section_name in section_names:
            section_data = ole.openstream(section_name).read()

            if compressed:
                try:
                    # HWP BodyText는 raw deflate인 경우가 많아서 -15 사용
                    section_data = zlib.decompress(section_data, -15)
                except Exception as e:
                    print(f"[압축 해제 실패] {hwp_path} / {section_name} / {e}")
                    continue

            section_text = extract_text_from_section(section_data)
            full_text_parts.append(section_text)

        full_text = clean_text("\n".join(full_text_parts))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")

        return True

    except Exception as e:
        print(f"[변환 실패] {hwp_path} / {e}")
        return False

    finally:
        ole.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hwp_files = list(DATA_DIR.rglob("*.hwp"))
    print(f"총 HWP 파일 수: {len(hwp_files)}")

    success = 0
    fail = 0

    for idx, hwp_path in enumerate(hwp_files, start=1):
        relative_path = hwp_path.relative_to(DATA_DIR)
        output_path = OUT_DIR / relative_path.with_suffix(".txt")

        print(f"[{idx}/{len(hwp_files)}] 변환 중: {relative_path}")

        ok = convert_one_hwp(hwp_path, output_path)

        if ok:
            success += 1
            print(f"  완료 → {output_path}")
        else:
            fail += 1
            print("  실패")

    print("\n변환 결과")
    print(f"성공: {success}")
    print(f"실패: {fail}")


if __name__ == "__main__":
    main()