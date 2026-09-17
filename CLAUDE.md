# dl-study

한국어 딥러닝/LLM/VLM 실습 학습 사이트. 구조와 규칙은 README.md 참고.

- 레슨 원본은 `lessons/*.md`뿐. 수정 후 반드시 `python build.py` 실행해서 `notebooks/`, `lessons.json` 재생성 후 함께 커밋.
- 새 용어가 레슨에 등장하면 `glossary.md`에도 추가 (`**용어 (영어)** — 설명` 형식, 빌드 불필요).
- 빌드 도구·프레임워크를 추가하지 말 것. 사이트는 `index.html` 한 파일.
- 레슨 코드는 실제로 실행해 검증할 것 (대상 환경: Colab T4, 최신 transformers/peft/datasets). 학습자는 초보자이므로 Trainer 같은 추상화보다 직접 짠 학습 루프를 유지.
