# LLM 평가 — "좋아졌다"를 숫자로 말하기

> ⏱ 60분 · T4 GPU 권장 (CPU로도 실행 가능, 느림) · 선수 지식: [LLM 다루기](#33-llm-inference)

**목표:** 파인튜닝·RAG·프롬프트를 바꿨을 때 정말 나아졌는지 **재는 방법**을 배웁니다. 자동 채점(정확 일치, 포함), LLM을 심사위원으로 쓰는 채점(LLM-as-a-judge), 그리고 각 방법의 함정. 평가가 없으면 모든 실험은 감상문이 됩니다.

## 왜 평가가 어려운가

분류는 정답이 하나라 맞았는지 세면 됩니다. LLM의 출력은 **자유 문장**입니다. "서울입니다", "대한민국의 수도는 서울이에요", "Seoul"이 모두 정답인데 문자열은 다 다릅니다. 그래서 과제 종류에 따라 채점 방법을 고릅니다.

| 과제 | 채점 방법 | 예 |
|---|---|---|
| 정답이 짧고 하나 | **정확 일치**(exact match) 또는 정규화 후 일치 | 산수, 분류 라벨, 예/아니오 |
| 정답이 문장 속에 있으면 됨 | **포함**(contains) | 사실 질문 ("서울"이 들어 있나) |
| 형식이 정해짐 | **파서로 검증** | JSON, 코드(실행해서 테스트 통과), 정규식 |
| 품질이 주관적 | **LLM 심사위원** 또는 사람 | 요약, 대화, 글쓰기 |
| 둘 중 어느 쪽이 나은가 | **쌍대 비교**(pairwise) | 모델 A vs B |

## 평가 세트 만들기

[내 프로젝트 시작하기](#50-your-project)에서 "학습 데이터보다 평가 세트를 먼저"라고 했습니다. 여기서는 세 종류 과제를 섞은 작은 세트를 만듭니다. 실제로는 50~200개를 만듭니다.

```python
import re, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

eval_set = [
    # (질문, 정답, 채점 방식)
    ("17 더하기 26은? 숫자만 답해.", "43", "exact"),
    ("9 곱하기 8은? 숫자만 답해.", "72", "exact"),
    ("100에서 37을 빼면? 숫자만 답해.", "63", "exact"),
    ("대한민국의 수도는 어디야?", "서울", "contains"),
    ("물의 화학식은?", "H2O", "contains"),
    ("지구에서 가장 큰 대양은?", "태평양", "contains"),
    ("1년은 몇 개월이야?", "12", "contains"),
    ("빛의 속도는 초속 약 몇 km야?", "30만", "contains"),
    ("다음 문장의 감정을 '긍정' 또는 '부정' 한 단어로 답해: 이 영화 정말 최고였어요!", "긍정", "exact"),
    ("다음 문장의 감정을 '긍정' 또는 '부정' 한 단어로 답해: 돈이 아까운 식당이었다.", "부정", "exact"),
    ("다음 문장의 감정을 '긍정' 또는 '부정' 한 단어로 답해: 배송이 빠르고 친절했습니다.", "긍정", "exact"),
    ("다음 문장의 감정을 '긍정' 또는 '부정' 한 단어로 답해: 두 번 다시 이용하지 않겠습니다.", "부정", "exact"),
    ("사과, 바나나, 포도를 JSON 배열로만 출력해.", '["사과", "바나나", "포도"]', "json"),
    ("이름이 '민수'이고 나이가 20인 사람을 name, age 키를 가진 JSON 객체로만 출력해.", '{"name": "민수", "age": 20}', "json"),
]
print(len(eval_set), "문항")
```

**코드 읽기**

- 문항마다 **채점 방식을 함께** 적습니다. 산수는 정확 일치, 사실 질문은 포함, JSON은 파싱 후 비교. 한 세트 안에 여러 방식을 섞어도 됩니다. 중요한 것은 문항을 만들 때 "무엇을 정답으로 칠지"를 **먼저** 정하는 것입니다.
- "숫자만 답해", "한 단어로만 답해" — 정확 일치 채점을 하려면 출력 형식을 프롬프트로 제한해야 합니다. 형식 제약 없이 정확 일치를 쓰면 정답인데 틀렸다고 채점되는 경우가 많아집니다.
- 이 세트는 학습에 쓰지 않습니다. 평가 세트는 끝까지 격리합니다.

## 자동 채점기

```python
def normalize(s):
    return re.sub(r"[\s\.\,\!\?。]", "", s).lower()          # 공백·문장부호 제거, 소문자

def score(pred, answer, method):
    if method == "exact":
        return float(normalize(pred) == normalize(answer))
    if method == "contains":
        return float(normalize(answer) in normalize(pred))
    if method == "json":
        try:
            start = pred.find("[") if answer.startswith("[") else pred.find("{")
            end = pred.rfind("]") if answer.startswith("[") else pred.rfind("}")
            return float(json.loads(pred[start:end + 1]) == json.loads(answer))
        except Exception:
            return 0.0

# 채점기 자체를 먼저 검증: 맞아야 할 것과 틀려야 할 것
assert score("43.", "43", "exact") == 1 and score("답은 43입니다", "43", "exact") == 0
assert score("대한민국의 수도는 서울입니다.", "서울", "contains") == 1
assert score('결과: ["사과", "바나나", "포도"] 입니다', '["사과", "바나나", "포도"]', "json") == 1
assert score("사과, 바나나, 포도", '["사과", "바나나", "포도"]', "json") == 0
print("채점기 검증 통과")
```

**코드 읽기**

- `normalize` — "43."과 "43"을 같게 보려고 문장부호·공백을 지웁니다. 어디까지 관대할지는 과제에 따라 정합니다. 너무 관대하면(예: 숫자만 추출) "43이 아니라 34입니다"도 맞다고 채점될 수 있습니다.
- `json` 방식 — 모델이 JSON 앞뒤에 말을 붙이는 일이 흔하므로 첫 `[`/`{`부터 마지막 `]`/`}`까지 잘라 파싱합니다. 파싱 실패는 0점. 형식 과제는 이렇게 **파서가 채점**하면 사람보다 정확합니다.
- 채점기를 `assert`로 검증 — 채점기가 틀리면 모든 실험 결과가 틀립니다. 맞아야 할 예와 틀려야 할 예를 각각 넣어 봅니다. 평가 코드도 코드이므로 테스트가 필요합니다.

## 두 모델을 같은 세트로 비교

```python
def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    return tok, model

def answer(tok, model, question, max_new_tokens=40):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def evaluate(name):
    tok, model = load(name)
    rows = []
    for q, a, m in eval_set:
        pred = answer(tok, model, q)
        rows.append({"q": q, "answer": a, "pred": pred, "method": m, "score": score(pred, a, m)})
    del model; torch.cuda.empty_cache()
    return rows

results = {name: evaluate(name) for name in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]}
for name, rows in results.items():
    by_method = {m: [r["score"] for r in rows if r["method"] == m] for m in ("exact", "contains", "json")}
    print(f"{name:32s} 전체 {sum(r['score'] for r in rows) / len(rows):.2f} | " + " | ".join(f"{m} {sum(v)/len(v):.2f}" for m, v in by_method.items()))
```

**코드 읽기**

- `do_sample=False` — 평가는 재현 가능해야 합니다. 샘플링을 켜면 실행할 때마다 점수가 달라져 비교가 안 됩니다. 다양성이 필요한 과제라면 같은 시드로 여러 번 돌려 평균합니다.
- `rows`에 질문·정답·예측·점수를 **모두** 저장 — 점수 하나만 남기면 "왜 틀렸는지"를 못 봅니다. 아래에서 틀린 것을 직접 읽습니다.
- `del model; torch.cuda.empty_cache()` — 두 모델을 동시에 올리지 않고 하나씩 평가해 GPU 메모리를 아낍니다.
- 방식별 점수를 따로 냅니다 — "전체 0.7"보다 "산수는 1.0인데 JSON은 0.3"이 다음에 무엇을 할지 알려줍니다.

```python
for name, rows in results.items():
    print(f"\n== {name} 이 틀린 문항")
    for r in rows:
        if r["score"] == 0:
            print(f"  Q: {r['q'][:40]}…  | 정답: {r['answer']}  | 모델: {r['pred'][:60]!r}")
```

틀린 것을 읽으면 두 종류가 보입니다. **정말 틀린 것**(산수 오답)과 **채점기가 못 잡은 것**(정답인데 형식이 달라 0점). 후자가 많으면 프롬프트나 채점기를 고칩니다. 이 구분을 하지 않고 점수만 보면 잘못된 결론을 내립니다.

## LLM을 심사위원으로: 주관적 품질 채점

요약이나 설명처럼 정답이 하나가 아닌 과제는 자동 채점이 안 됩니다. 사람이 채점하는 것이 정석이지만 비용이 크므로, **더 큰 LLM에게 루브릭을 주고 채점**시키는 방법을 널리 씁니다.

```python
judge_tok, judge = load("Qwen/Qwen2.5-1.5B-Instruct")     # 심사위원은 평가 대상보다 큰 모델
small_tok, small = load("Qwen/Qwen2.5-0.5B-Instruct")

open_questions = ["광합성을 초등학생에게 두 문장으로 설명해줘.", "커피와 차의 차이를 세 가지 말해줘.", "왜 하늘은 파란지 한 문단으로 설명해줘."]
JUDGE_SYSTEM = """너는 엄격한 채점관이다. 질문의 요구(내용, 길이, 형식)를 답변이 얼마나 충족하는지 1~5점으로 채점한다.
5: 정확하고 요구를 모두 지킴 / 3: 대체로 맞지만 요구 일부를 어김 / 1: 틀렸거나 질문과 무관하거나 답하지 않음
출력 형식은 반드시 다음 두 줄뿐이다.
점수: <1~5 숫자>
이유: <한 문장>"""

def judge_score(question, response):
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"[질문]
{question}

[답변]
{response}"}]
    prompt = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = judge.generate(**inputs, max_new_tokens=60, do_sample=False)
    verdict = judge_tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"점수\s*[:：]\s*([1-5])", verdict)
    return (int(m.group(1)) if m else None), verdict

for q in open_questions:
    resp = answer(small_tok, small, q, max_new_tokens=120)
    s, verdict = judge_score(q, resp)
    print(f"Q: {q}
답변: {resp[:100]}…
심사: {s}점 — {verdict.replace(chr(10), ' | ')[:90]}
")
```

**코드 읽기**

- **루브릭을 시스템 프롬프트에** — 점수의 기준을 글로 명시하고, 사용자 메시지에는 질문과 답변만 넣습니다. 기준을 사용자 메시지 끝에 붙이면 작은 모델은 답변 내용에 끌려가 형식을 자주 무시합니다(이 레슨을 만들 때 실제로 그랬습니다). "역할과 규칙은 system, 채점 대상은 user"로 나누면 형식 준수율이 크게 오릅니다.
- **루브릭** — 점수의 기준을 글로 명시합니다. "좋은 답에 높은 점수"처럼 모호하면 심사가 일관되지 않습니다. 기준을 구체적으로 쓸수록(길이, 형식, 사실성) 사람 평가와의 일치도가 올라갑니다.
- `'점수: N' 형식으로 첫 줄에` — 심사 결과를 **파싱 가능한 형식**으로 강제하고 정규식으로 뽑습니다. 파싱에 실패하면 `None`으로 두고 나중에 셉니다. 실패율이 높으면 루브릭의 형식 지시를 고칩니다.
- 심사 결과 예(1.5B 심사위원): 요구를 어긴 장황한 답에 3점, 거절한 답에 1점을 줍니다. 대체로 방향은 맞지만, 좋은 답에도 3점을 주는 등 후하지도 정확하지도 않습니다. `Qwen/Qwen2.5-3B-Instruct`(T4에서 `dtype=torch.float16`)로 바꾸면 눈에 띄게 나아집니다.
- 심사위원이 평가 대상보다 커야 하는 이유 — 작은 모델이 큰 모델을 채점하면 틀린 답을 맞다고 하기 쉽습니다. 실무에서는 GPT-4급 API 모델이나 전용 심사 모델을 씁니다. 1.5B는 실습용 최소치이며 신뢰도가 높지 않습니다. 이 레슨의 목적은 **방법을 익히는 것**이고, 실제 프로젝트에서는 더 큰 심사 모델과 사람 검증이 필요합니다.

## LLM 심사의 함정

```python
# 1) 같은 답을 두 번 채점: 점수가 같은가? (일관성)
resp = answer(small_tok, small, open_questions[0], max_new_tokens=120)
print("같은 답변 2회 채점:", [judge_score(open_questions[0], resp)[0] for _ in range(2)])

# 2) 길이 편향: 내용은 같은데 더 긴 답에 점수를 더 주는가?
short = "광합성은 식물이 햇빛으로 양분을 만드는 과정이에요. 물과 공기를 재료로 써서 산소도 내보내요."
long = short + " 이 과정은 잎의 엽록체에서 일어나며, 지구 생태계의 거의 모든 에너지가 여기서 시작됩니다. 정말 놀라운 일이지요!"
print("두 문장 답:", judge_score(open_questions[0], short)[0], " / 늘린 답(요구 위반):", judge_score(open_questions[0], long)[0])   # 요구를 어긴 긴 답이 같거나 높은 점수면 심사위원이 길이 요구를 못 잡는 것
```

**코드 읽기**

- **일관성** — greedy 디코딩이라 같은 입력에는 같은 점수가 나와야 합니다. 샘플링을 켜면 달라지므로 심사에도 `do_sample=False`를 씁니다.
- **길이 편향** — LLM 심사위원은 길고 그럴듯한 답을 선호하는 경향이 있습니다. 두 문장이라는 요구를 어긴 긴 답이 같거나 더 높은 점수를 받는다면(1.5B는 둘 다 3점을 줍니다) 심사위원이 길이 요구를 채점에 반영하지 못하는 것입니다. 이런 검사를 **심사위원의 평가**라고 하며, 사람이 채점한 소수 샘플과 심사 점수의 상관을 재는 것이 표준입니다.
- 그 밖의 알려진 편향: 쌍대 비교에서 **앞에 나온 답을 선호**(position bias) → 순서를 바꿔 두 번 채점하고 평균. **자기 선호**(self-preference) → 평가 대상과 다른 계열의 심사 모델 사용.

## 공개 벤치마크를 볼 때

MMLU, HumanEval, KMMLU 같은 벤치마크 점수는 모델을 고를 때 참고가 되지만 다음을 알아야 합니다.

- **오염(contamination):** 벤치마크 문제가 사전학습 데이터에 섞여 있으면 점수가 부풀려집니다. 새 벤치마크일수록, 비공개일수록 신뢰도가 높습니다.
- **내 과제와의 거리:** 수능형 객관식 점수가 "우리 고객 문의 응대" 능력을 말해 주지 않습니다. **내 평가 세트**가 최종 기준입니다.
- **프롬프트·채점 방식에 따라 점수가 크게 변합니다.** 다른 논문의 숫자를 그대로 비교하지 말고 같은 조건에서 직접 재세요.

## 핵심 정리

- 과제 종류에 따라 채점기를 고릅니다: 정확 일치 / 포함 / 파서 / LLM 심사 / 사람.
- 채점기부터 `assert`로 검증하고, 점수 옆에 **틀린 샘플**을 항상 읽습니다.
- 평가는 greedy로 재현 가능하게, 평가 세트는 학습에서 격리합니다.
- LLM 심사위원은 루브릭·형식 강제·편향 검사가 필수이며, 사람 채점과의 일치도로 신뢰도를 확인합니다.
- 공개 벤치마크는 참고일 뿐, 내 평가 세트가 기준입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 파인튜닝 후 정확 일치 점수가 떨어졌는데 틀린 샘플을 보니 답은 맞고 형식만 달라졌습니다. 어떻게 하나요?</summary>

모델이 나빠진 것이 아니라 채점기가 형식 변화를 못 잡는 것입니다. 채점기를 형식에 덜 민감하게 고치거나(정규화 강화, 파서 사용), 프롬프트로 형식을 다시 고정합니다. 어느 쪽이든 "채점기 변경"은 기록해 두고 이전 실험도 같은 채점기로 다시 채점해야 비교가 됩니다.

</details>

<details><summary>Q2. 쌍대 비교에서 A와 B의 순서를 바꿔 두 번 채점하는 이유는?</summary>

LLM 심사위원은 먼저 제시된 답을 선호하는 위치 편향이 있습니다. 순서를 바꿔 두 번 채점해 결과가 뒤집히면 "동점"으로 처리하고, 두 번 모두 같은 쪽을 고를 때만 승패로 칩니다.

</details>

<details><summary>Q3. 평가 세트 14문항으로 두 모델의 점수가 0.71과 0.79입니다. 뒤 모델이 더 낫다고 말할 수 있나요?</summary>

문항 1개 차이입니다. 이 정도 차이는 우연일 수 있습니다. 문항을 100개 이상으로 늘리거나, 여러 시드·프롬프트 변형으로 반복해 차이가 일관되는지 확인해야 합니다. 작은 평가 세트의 점수 차이는 "신호"일 뿐 "결론"이 아닙니다.

</details>

## 직접 고쳐보기

1. `eval_set`에 내 과제 문항 20개를 추가하고 채점 방식을 정하세요. 만들면서 "정답이 하나로 정해지지 않는 문항"이 얼마나 많은지 세어 보세요.
2. [LLM 파인튜닝](#34-llm-finetune)에서 만든 "냥" 어댑터를 붙인 모델을 이 세트로 평가해 보세요. 말투를 바꾸느라 다른 능력이 얼마나 떨어졌나요? (catastrophic forgetting의 정량 측정)
3. `normalize`를 더 관대하게(숫자만 추출) 바꿔 보세요. 어떤 문항이 새로 "맞음"이 되고, 그중 실제로 틀린 답이 있나요?
4. 쌍대 비교 심사를 구현하세요: 같은 질문에 대한 0.5B와 1.5B의 답을 A/B로 제시하고 "A 또는 B"만 답하게 한 뒤, 순서를 바꿔 다시 물어 일치하는지 확인하세요.
5. (도전) 심사 점수와 **내 판단**을 비교하세요: 답변 10개를 직접 1~5점으로 채점한 뒤 심사 점수와 상관계수(`numpy.corrcoef`)를 구하세요. 0.5 미만이면 이 심사위원은 쓸 수 없습니다.

<details><summary>힌트와 예상 결과 — 먼저 스스로 해 본 뒤 펼치세요</summary>

1. 만들다 보면 "정답이 하나인 문항"이 생각보다 적다는 것을 알게 됩니다. 그 문항들은 포함 채점이나 심사로 넘기고, 정확 일치 문항은 프롬프트로 형식을 고정하세요.
2. "냥" 모델은 산수·사실 문항에서 점수가 떨어지고(특히 형식 지시를 무시하고 "~냥"을 붙여 정확 일치 실패), JSON 문항은 크게 망가집니다. 이 하락폭이 catastrophic forgetting의 정량값입니다.
3. 숫자만 추출하면 "17 + 26 = 43"이 맞음이 되지만, "93은 아니고 63입니다" 같은 답에서 첫 숫자를 잘못 뽑아 틀린 답을 맞다고 할 수 있습니다. 관대함에는 대가가 있습니다. 어느 쪽 오류가 더 해로운지로 정합니다.
4. 프롬프트에 "[답변 A] ... [답변 B] ... 더 나은 쪽을 A 또는 B 한 글자로만 답해"를 넣고, 순서를 바꿔 두 번 물어 일치하는 경우만 승패로 칩니다. 1.5B 심사위원은 위치 편향이 커서 뒤집히는 비율이 30% 이상 나올 수 있습니다.
5. `np.corrcoef(내점수, 심사점수)[0, 1]`. 1.5B 심사위원은 0.3~0.5 정도로 낮게 나올 가능성이 큽니다. 실제 프로젝트에서는 더 큰 심사 모델(API 모델 등)로 0.7 이상을 확보한 뒤에 씁니다.

</details>
