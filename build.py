"""lessons/*.md -> notebooks/*.ipynb + lessons.json.  실행: python build.py"""
import json
import pathlib
import re

root = pathlib.Path(__file__).parent
(root / "notebooks").mkdir(exist_ok=True)
FENCE = re.compile(r"```python\n(.*?)```\n?", re.S)


def cells(md):
    # re.split with one capture group: even idx = markdown, odd idx = python code
    for i, part in enumerate(FENCE.split(md)):
        if part.strip():
            yield {"cell_type": "code", "metadata": {}, "source": part.rstrip(), "outputs": [], "execution_count": None} if i % 2 \
                else {"cell_type": "markdown", "metadata": {}, "source": part.strip()}


index = []
for f in sorted((root / "lessons").glob("*.md")):
    md = f.read_text(encoding="utf-8")
    nb = {"cells": list(cells(md)), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    (root / "notebooks" / f"{f.stem}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    index.append({"id": f.stem, "title": md.splitlines()[0].lstrip("# ").strip(),
                  "code": any(c["cell_type"] == "code" for c in nb["cells"])})

(root / "lessons.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
assert index and all(x["title"] for x in index), "레슨 첫 줄은 '# 제목' 이어야 합니다"
print(f"{len(index)} lessons built")
