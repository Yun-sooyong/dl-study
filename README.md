# 직접 만들며 배우는 딥러닝 · LLM · VLM

텐서 → MLP → CNN → 미니 GPT → LLM LoRA 파인튜닝 → 미니 VLM → VLM 파인튜닝.
각 레슨은 사이트에서 읽고, 버튼 하나로 Colab(무료 GPU)에서 실행합니다.

## 구조

```
index.html      사이트 전체 (빌드 도구 없음. marked + highlight.js CDN)
lessons/*.md    레슨 원본 ← 여기만 고치면 됩니다
glossary.md     용어 사전 (`**용어** — 설명` 한 문단이 항목 하나, `##`가 분류)
build.py        lessons/*.md → notebooks/*.ipynb + lessons.json 생성
notebooks/      Colab용 노트북 (생성물, 직접 수정 금지)
lessons.json    목차 (생성물)
```

## 다른 컴퓨터에서 이어서 작업하기

```bash
git clone https://github.com/Yun-sooyong/dl-study.git
cd dl-study
python -m http.server 8000     # http://localhost:8000 에서 미리보기
```

레슨을 고치거나 추가한 뒤:

```bash
python build.py                # 노트북과 목차 재생성 (파이썬 표준 라이브러리만 사용)
git add -A && git commit -m "레슨 수정" && git push
```

## 레슨 작성 규칙

- 파일명 `NN-이름.md`, 첫 줄은 `# 제목` (목차에 그대로 표시됨)
- ` ```python ` 블록만 노트북의 코드 셀이 됩니다. 그 외는 전부 마크다운 셀.
- 코드 셀은 위에서 아래로 순서대로 실행했을 때 동작해야 합니다.
- 레슨 끝에는 **직접 고쳐보기** 과제를 둡니다.
