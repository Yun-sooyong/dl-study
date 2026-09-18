# 직접 만들며 배우는 머신러닝 · 딥러닝 · LLM · VLM

사이트: https://yun-sooyong.github.io/dl-study/

0부(환경, 필요한 수학) → 머신러닝 기초 → 딥러닝(역전파 직접 구현 포함) → LLM(토크나이저, 미니 GPT, 사전학습 실전, 추론, LoRA 파인튜닝, RAG, DPO, 평가) → VLM(CLIP, 미니 VLM, VLM 파인튜닝) → 프로젝트(가이드, 종합 과제와 채점표, 한계와 책임)까지 26개 레슨,
각 레슨을 사이트에서 읽고 버튼 하나로 Colab(무료 GPU)에서 실행합니다. 모든 레슨 코드는 실제로 실행해 검증했습니다.

## 구조

```
index.html      사이트 전체 (빌드 도구 없음. marked + highlight.js CDN)
lessons/*.md    레슨 원본 ← 여기만 고치면 됩니다. 파일명 첫 숫자가 파트 (0=시작, 1=ML, 2=DL, 3=LLM, 4=VLM, 5=프로젝트)
glossary.md     용어 사전. `**용어 / 별칭 (english)** — 설명` 한 문단이 항목 하나, `##`가 분류.
                레슨 본문에 처음 나오는 용어는 자동으로 툴팁이 붙습니다.
build.py        lessons/*.md → notebooks/*.ipynb + lessons.json 생성, 레슨 간 링크 검사
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
git add -A && git commit -m "레슨 수정" && git push    # push하면 1~2분 뒤 사이트에 반영
```

## 레슨 작성 규칙

- 파일명 `NN-이름.md` (첫 숫자 = 파트, build.py의 `PARTS`), 첫 줄은 `# 제목` (목차에 그대로 표시됨)
- 둘째 문단은 `> ⏱ 소요시간 · 실행 환경` (사이트에서 칩으로 표시됨)
- 순서: **목표** → 본문 → `## 핵심 정리` → `## 스스로 점검` (`<details><summary>질문</summary>` 퀴즈) → `## 직접 고쳐보기`.
  이 세 제목은 사이트가 자동으로 색 박스로 꾸밉니다. `<summary>` 안에서는 마크다운 대신 `<b>`, `<code>`를 씁니다.
- 코드 블록 뒤에는 `**코드 읽기**` 문단 + 목록으로 해설을 답니다 (사이트가 해설 박스로 꾸밈).
  각 항목은 `` `함수/클래스` — 무엇을 하나. 왜 이걸 쓰나(대안과 비교) `` 형식으로, "무엇"보다 **"왜"** 를 씁니다.
- ` ```python ` 블록만 노트북의 코드 셀이 됩니다. 그 외는 전부 마크다운 셀. 위에서 아래로 순서대로 실행했을 때 동작해야 합니다.
- 다른 레슨은 `[이름](#파일id)`로 링크 (없는 id면 build.py가 에러를 냅니다)
- 그래프 제목·라벨은 영어로 (Colab matplotlib에 한글 폰트가 없음)
- `## 직접 고쳐보기` 목록 뒤에 `<details><summary>힌트와 예상 결과 …</summary>` 블록으로 문항별 예상 결과를 적습니다(실제로 돌려 본 값 기준).
- 새 용어가 나오면 `glossary.md`에 추가
