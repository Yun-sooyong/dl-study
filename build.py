"""lessons/*.md -> notebooks/*.ipynb + lessons.json.  실행: python build.py"""
import json
import pathlib
import re

SITE = "https://yun-sooyong.github.io/dl-study/"
PARTS = {  # 파일명 첫 글자 -> 파트 이름
    "0": "시작",
    "1": "1부 · 머신러닝 기초",
    "2": "2부 · 딥러닝 기초",
    "3": "3부 · LLM",
    "4": "4부 · VLM",
    "5": "5부 · 내 프로젝트로",
}
root = pathlib.Path(__file__).parent
(root / "notebooks").mkdir(exist_ok=True)
FENCE = re.compile(r"```python\n(.*?)```\n?", re.S)


def cells(md):
    md = md.replace("](#", f"]({SITE}#")  # 사이트 내부 링크는 노트북에서 절대 주소로
    # re.split with one capture group: even idx = markdown, odd idx = python code
    for i, part in enumerate(FENCE.split(md)):
        if part.strip():
            yield {"cell_type": "code", "metadata": {}, "source": part.rstrip(), "outputs": [], "execution_count": None} if i % 2 \
                else {"cell_type": "markdown", "metadata": {}, "source": part.strip()}


index = []
ids = {f.stem for f in (root / "lessons").glob("*.md")} | {"glossary"}
for f in sorted((root / "lessons").glob("*.md")):
    md = f.read_text(encoding="utf-8")
    broken = [t for t in re.findall(r"\]\(#([^)]+)\)", md) if t not in ids]
    assert not broken, f"{f.name}: 없는 레슨으로 가는 링크 {broken}"
    nb = {"cells": list(cells(md)), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    (root / "notebooks" / f"{f.stem}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    index.append({"id": f.stem, "title": md.splitlines()[0].lstrip("# ").strip(), "part": PARTS[f.stem[0]],
                  "code": any(c["cell_type"] == "code" for c in nb["cells"])})

for stale in (root / "notebooks").glob("*.ipynb"):  # 이름이 바뀌거나 삭제된 레슨의 노트북 정리
    if stale.stem not in ids:
        stale.unlink()

glossary = (root / "glossary.md").read_text(encoding="utf-8")
broken = [t for t in re.findall(r"\]\(#([^)]+)\)", glossary) if t not in ids]
assert not broken, f"glossary.md: 없는 레슨으로 가는 링크 {broken}"

(root / "lessons.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
assert index and all(x["title"] for x in index), "레슨 첫 줄은 '# 제목' 이어야 합니다"
print(f"{len(index)} lessons built")
