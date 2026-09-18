# 사전학습 LLM 다루기 — 생성, 샘플링, 프롬프트

> ⏱ 60분 · T4 GPU 권장 (CPU로도 실행 가능, 느림)

**목표:** 실제 LLM을 내려받아 `generate()` 한 줄 뒤에서 벌어지는 일을 **직접 손으로** 해 봅니다. 로짓 → 확률 → 샘플링, temperature와 top-p, KV 캐시, 채팅 템플릿과 프롬프트 기법까지. 모델을 고치기 전에 먼저 잘 다룰 줄 알아야 합니다.

## 모델 불러오기

[미니 GPT](#31-mini-gpt)와 같은 구조, 5억 개 파라미터, 수조 토큰으로 사전학습된 모델입니다.

```python
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 파라미터, 층 {model.config.num_hidden_layers}개, "
      f"임베딩 {model.config.hidden_size}차원, 어휘 {model.config.vocab_size}개")
```

**코드 읽기**

- 왜 Qwen2.5-0.5B-Instruct인가: ① 5억 파라미터라 무료 T4는 물론 CPU에서도 돌아가고, ② 한국어를 쓸 만큼 하며, ③ Instruct 버전이라 대화 형식을 이해합니다. 원리를 배우는 데는 큰 모델이 필요 없고, 작은 모델의 실패를 보는 것도 공부입니다. 더 큰 모델(1.5B, 3B, 7B)로 바꿔도 코드는 한 글자도 안 바뀝니다.
- `AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)` — "Causal LM"은 앞 토큰으로 다음 토큰을 예측하는 모델(=GPT 구조). `Auto`가 모델 종류를 알아서 판별하고, 가중치 파일(약 1GB)을 내려받아 끼웁니다. `dtype=torch.float32`는 32비트 실수로 올리라는 뜻: 0.5B는 fp32로도 2GB뿐이라 여유 있고, 뒤의 학습 레슨에서 수치 안정성이 좋습니다. 큰 모델은 `torch.bfloat16`으로 올려 메모리를 절반으로 줄입니다.
- `.to(device).eval()` — GPU로 옮기고 평가 모드로. 추론만 할 때는 항상 `eval()`.
- `model.config` — 모델의 설계 값(층 수, 임베딩 차원, 어휘 크기, 헤드 수 등)을 담은 객체. `config.json` 파일 내용입니다. [미니 GPT](#31-mini-gpt)의 `n_layer`, `n_embd`, `vocab_size`가 여기서는 24, 896, 151936입니다.
## forward 한 번 = 다음 토큰의 확률분포

```python
text = "대한민국의 수도는"
ids = tok(text, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    logits = model(ids).logits
print(ids.shape, "→", logits.shape)          # [1, 토큰 수] → [1, 토큰 수, 어휘 크기]

probs = F.softmax(logits[0, -1], dim=-1)     # 마지막 위치 = "다음에 올 토큰"에 대한 예측
top = probs.topk(5)
for p, i in zip(top.values, top.indices):
    print(f"{p.item():.3f}  {tok.decode(i)!r}")
```

**코드 읽기**

- `tok(text, return_tensors="pt")` — 문자열을 토큰 번호 텐서로. `return_tensors="pt"`는 파이썬 리스트 대신 PyTorch 텐서(`[1, 토큰 수]`, 배치 차원 포함)로 달라는 뜻. 결과에는 `input_ids`와 `attention_mask`가 들어 있습니다.
- `model(ids).logits` — Hugging Face 모델의 출력은 여러 값을 담은 객체이고, `.logits`가 `[배치, 토큰 수, 어휘]` 텐서입니다. [미니 GPT](#31-mini-gpt)의 `forward`가 돌려주던 `logits`와 같은 것입니다.
- `logits[0, -1]` — 배치 0번, **마지막 위치**. 각 위치의 로짓은 "그 위치까지 봤을 때 다음 토큰" 예측이므로, 문장 끝에 이어질 토큰은 마지막 위치를 봐야 합니다. 앞 위치들의 예측은 학습 때 쓰이고(모든 위치에서 손실 계산), 생성 때는 마지막만 씁니다.
- `F.softmax(..., dim=-1)` — 15만 개 로짓을 합이 1인 확률로. `topk(5)`가 큰 순서로 5개의 값과 인덱스를 돌려줍니다.

`서울`이 1등이 아니어서 놀랐나요? 모델은 "정답"을 예측하는 것이 아니라 **학습한 글들에서 이 뒤에 흔히 이어지던 것**을 예측합니다. "대한민국의 수도는?" 하고 퀴즈가 이어지는 글이 많았던 것입니다.

모델이 하는 일은 이것이 전부입니다. **15만 개 토큰 각각이 다음에 올 확률**을 내놓는 것. 문장을 만드는 것은 이 확률에서 토큰을 하나 고르고, 붙이고, 다시 묻는 **반복문**입니다.

## 생성 루프 직접 만들기

```python
@torch.no_grad()
def greedy(text, n=20):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    for _ in range(n):
        next_id = model(ids).logits[0, -1].argmax()            # 가장 확률 높은 토큰 하나
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)      # 뒤에 붙이고 다시
    return tok.decode(ids[0])

print(greedy("대한민국의 수도는"))
```

## 샘플링: temperature, top-k, top-p

항상 1등만 고르면(greedy) 결과가 매번 같고, 같은 말을 반복하기 쉽습니다. 그래서 보통은 확률에 따라 **뽑습니다**. 뽑는 방식을 조절하는 손잡이가 세 개 있습니다.

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature                      # T<1: 분포가 뾰족해짐(안전), T>1: 평평해짐(모험)
    if top_k:                                          # 상위 k개 밖은 버림
        kth = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p:                                          # 확률을 큰 것부터 더해 top_p를 넘는 지점까지만 남김
        sorted_p, sorted_i = probs.sort(descending=True)
        keep = sorted_p.cumsum(0) - sorted_p < top_p
        probs = torch.zeros_like(probs).scatter(0, sorted_i[keep], sorted_p[keep])
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1)

@torch.no_grad()
def sample(text, n=30, **kw):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    for _ in range(n):
        ids = torch.cat([ids, sample_next(model(ids).logits[0, -1], **kw).view(1, 1)], dim=1)
    return tok.decode(ids[0])

torch.manual_seed(0)
prompt = "오늘 아침에 일어나 보니"
for kw in [dict(temperature=0.3), dict(temperature=1.0), dict(temperature=2.0), dict(temperature=1.0, top_p=0.9)]:
    print(kw, "\n ", sample(prompt, **kw).replace("\n", " "), "\n")
```

**코드 읽기**

- `logits / temperature` — 로짓을 나눈 뒤 softmax를 하면 분포의 뾰족함이 바뀝니다. 0.5로 나누면 로짓 차이가 2배로 벌어져 1등이 더 압도적이 되고, 2로 나누면 차이가 절반이 되어 하위 후보도 뽑힙니다. 확률 자체를 건드리는 것보다 로짓에서 조절하는 것이 수학적으로 깔끔합니다.
- `top_k`: `logits.topk(top_k).values[-1]` — k번째로 큰 로짓 값. 그보다 작은 로짓을 `-inf`로 바꾸면 softmax 후 확률이 0이 되어 후보에서 빠집니다.
- `top_p`: `probs.sort(descending=True)` → `cumsum` — 확률을 큰 것부터 누적해서 `top_p`(0.9)에 도달할 때까지의 후보만 남깁니다. `cumsum(0) - sorted_p < top_p`는 "이 후보를 더하기 **전**의 누적이 0.9 미만"이라는 조건이라, 0.9를 넘기게 만드는 후보까지 포함합니다(그래야 후보가 최소 하나는 남습니다). `scatter`로 원래 순서의 자리에 확률을 되돌려 놓고, 합이 1이 되게 다시 나눕니다.
- `torch.multinomial(probs, 1)` — 최종 확률에 따라 하나를 뽑습니다. 세 손잡이는 모두 "뽑기 전에 확률을 어떻게 다듬을지"이고, 뽑는 행위는 이 한 줄입니다.
- `sample(prompt, **kw)` — `**kw`로 키워드 인자를 그대로 넘겨서 `sample(prompt, temperature=0.3)`처럼 설정을 바꿔 가며 부를 수 있게 했습니다. 실험용 함수를 만들 때 유용한 파이썬 패턴입니다.
- `model.generate(...)`의 인자 `temperature`, `top_k`, `top_p`, `do_sample`이 정확히 여기서 구현한 것과 같습니다. 직접 짜 본 이유는 라이브러리의 이 인자들이 **무엇을 하는지 알고 쓰기** 위해서입니다.

- **temperature 낮게(0~0.5):** 사실 질문, 코드, 추출 등 정답이 있는 작업
- **temperature 높게(0.8~1.2) + top-p 0.9:** 글쓰기, 아이디어 등 다양성이 필요한 작업
- temperature 2.0의 결과가 왜 무너지는지 확률분포의 관점에서 설명할 수 있나요?

## KV 캐시: 생성이 빠른 이유

위의 루프는 토큰을 하나 추가할 때마다 **문장 전체를 처음부터 다시** 계산합니다. 하지만 causal 어텐션에서는 앞 토큰들의 Key, Value가 변하지 않으므로 저장해 두고(**KV 캐시**) 새 토큰 하나만 계산하면 됩니다. `generate()`는 이것을 기본으로 사용합니다.

```python
inputs = tok("인공지능의 역사를 설명해줘.", return_tensors="pt").to(device)
for use_cache in (False, True):
    t = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=60, min_new_tokens=60, do_sample=False, use_cache=use_cache)
    print(f"use_cache={use_cache}: {time.time() - t:.2f}초")
```

문장이 길어질수록 차이가 커집니다. 대신 캐시는 메모리를 차지하며, 긴 컨텍스트에서 GPU 메모리가 부족해지는 주된 원인입니다.

**코드 읽기**

- `model.generate(**inputs, max_new_tokens=60, ...)` — `**inputs`는 토크나이저가 만든 `input_ids`와 `attention_mask`를 키워드 인자로 풀어 넣는 문법. `max_new_tokens`는 **새로** 만들 토큰 수의 상한(프롬프트 길이 제외). `min_new_tokens=60`을 같이 준 이유는 두 실행의 생성 길이를 같게 맞춰 공정하게 시간을 재기 위해서입니다(모델이 일찍 끝내면 비교가 안 됨).
- `do_sample=False` — 샘플링 없이 greedy. 시간 비교에서 결과가 같도록.
- `use_cache=False` — KV 캐시를 끄면 위의 `greedy`처럼 매번 전체를 다시 계산합니다. 기본값은 `True`이며 끌 이유는 실험 말고는 없습니다.
- `time.time()` 차이로 시간 재기 — 간단한 벤치마크는 이걸로 충분합니다. GPU에서는 비동기 실행 때문에 정확히 재려면 `torch.cuda.synchronize()`가 필요하지만, 몇 배 차이를 보는 데는 문제없습니다.

## 채팅 템플릿: Instruct 모델에게 말 거는 법

이 모델은 사전학습 후 **대화 형식의 데이터로 추가 학습**된 Instruct 모델입니다. 그 형식(특수 토큰으로 역할을 구분)에 맞춰 입력해야 제 실력이 나옵니다.

```python
def chat(messages, max_new_tokens=120, **kw):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

q = "김치찌개 끓이는 법을 세 단계로 알려줘."
print("── 템플릿 없이 그냥 이어 쓰기 ──\n", greedy(q, 60))
print("\n── 채팅 템플릿 사용 ──\n", chat([{"role": "user", "content": q}]))
```

**코드 읽기**

- `messages = [{"role": ..., "content": ...}, ...]` — 대화를 표현하는 표준 형식. `role`은 `system`(규칙), `user`(사용자), `assistant`(모델). OpenAI API와 같은 형식이라 다른 모델·서비스로 옮겨도 그대로 씁니다.
- `tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` — 이 리스트를 모델이 학습한 **특수 토큰 형식의 문자열**로 바꿉니다(`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). 형식은 모델마다 다르며 토크나이저 설정 파일에 들어 있으므로, 직접 문자열을 조립하지 말고 이 함수를 써야 합니다. `add_generation_prompt=True`는 끝에 `<|im_start|>assistant\n`을 붙여 "이제 네 차례"라고 알리는 것. 이것을 빼면 모델이 사용자 발화를 이어 쓰려 합니다. `tokenize=False`는 토큰이 아닌 문자열로 달라는 뜻이고, 그다음 줄에서 따로 토큰화합니다(중간 결과를 눈으로 확인하기 위해).
- `out[0, inputs.input_ids.shape[1]:]` — `generate`는 프롬프트 + 생성 토큰을 통째로 돌려주므로, 프롬프트 길이만큼 잘라 **새로 생성된 부분만** 디코딩합니다.
- `skip_special_tokens=True` — `<|im_end|>` 같은 특수 토큰을 출력에서 뺍니다.
- 템플릿 없이 넣었을 때의 결과 — 모델은 질문을 "이어 쓸 글의 첫 줄"로 봅니다. 블로그 글 같은 것이 이어지거나 같은 말을 반복합니다. 학습 때 본 형식과 다르기 때문입니다. **Instruct 모델에 템플릿 없이 물어보고 "성능이 나쁘다"고 결론 내리는 것**이 초보자의 흔한 실수입니다.
## 프롬프트로 행동 바꾸기

가중치를 건드리지 않고 **입력만으로** 모델의 행동을 바꾸는 기술입니다. 파인튜닝을 고려하기 전에 항상 먼저 시도합니다.

**시스템 프롬프트** — 역할과 규칙을 정합니다.

```python
q = "블랙홀이 뭐야?"
print(chat([{"role": "system", "content": "너는 유치원 선생님이다. 다섯 살 아이에게 말하듯 두 문장으로만 답한다."},
            {"role": "user", "content": q}]))
```

**퓨샷(few-shot)** — 예시를 몇 개 보여주면 형식을 따라 합니다. 모델이 예시에서 패턴을 읽어내는 능력(in-context learning)을 이용합니다.

```python
few_shot = [
    {"role": "system", "content": "리뷰의 감정을 분류한다. 반드시 '긍정' 또는 '부정' 한 단어로만 답한다."},
    {"role": "user", "content": "배송도 빠르고 품질도 좋아요"}, {"role": "assistant", "content": "긍정"},
    {"role": "user", "content": "한 번 쓰고 고장났습니다"}, {"role": "assistant", "content": "부정"},
]
for review in ["가격 대비 정말 만족합니다", "다시는 안 삽니다", "생각보다 별로네요"]:
    print(review, "→", chat(few_shot + [{"role": "user", "content": review}], max_new_tokens=5))
```

**코드 읽기**

- 시스템 프롬프트는 `messages`의 **첫 번째** 항목에 `role: "system"`으로 넣습니다. 템플릿이 이를 대화 맨 앞에 배치하고, 모델은 사전학습·SFT 과정에서 "이 자리의 지시를 따르라"고 배웠습니다. Qwen은 시스템 프롬프트를 안 주면 기본 문구("You are Qwen…")를 자동으로 넣습니다.
- 퓨샷은 `user`/`assistant` 쌍을 **가짜 대화 기록**으로 앞에 붙이는 것입니다. 모델 입장에서는 "이미 이렇게 대답해 온 대화"를 보는 셈이라 형식을 따라 합니다. 예시의 답을 우리가 써 넣었다는 점이 핵심입니다. 모델이 실제로 한 말이 아닙니다.
- `few_shot + [...]` — 리스트 연결. 매 질문마다 예시 리스트를 복사해 새 질문을 붙입니다. 예시 자체는 바뀌지 않습니다.
- `max_new_tokens=5` — 답이 "긍정"/"부정" 한 단어여야 하므로 생성을 짧게 끊습니다. 형식을 강제하는 가장 값싼 방법은 **생성 길이 제한**입니다.
## 작은 모델의 한계 확인하기

```python
for q in ["세종대왕이 맥북을 던진 사건에 대해 설명해줘.", "37 곱하기 48은?"]:
    print(f"Q: {q}\nA: {chat([{'role': 'user', 'content': q}], max_new_tokens=80)}\n")
```

모델은 "모른다"고 하기보다 **그럴듯한 다음 토큰**을 이어 갑니다. 이것이 환각(hallucination)입니다. 모델이 클수록 줄지만 사라지지는 않습니다. 대응 방법은 뒤의 [임베딩과 RAG](#35-embeddings-rag) 레슨에서 다룹니다.

## 핵심 정리

- LLM의 forward는 **다음 토큰의 확률분포**를 내놓고, 생성은 "뽑고 붙이고 반복"하는 루프입니다.
- temperature는 분포의 뾰족함을, top-k/top-p는 후보의 범위를 조절합니다.
- KV 캐시는 앞 토큰의 계산을 재사용해 생성을 빠르게 합니다(메모리와 교환).
- Instruct 모델에는 반드시 채팅 템플릿을 씁니다.
- 시스템 프롬프트와 퓨샷만으로도 많은 과제가 해결됩니다. **프롬프트 → RAG → 파인튜닝** 순서로 시도하세요.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. temperature를 0에 가깝게 하면 어떤 방식과 같아지나요?</summary>

로짓을 아주 작은 수로 나누면 1등과 나머지의 차이가 극단적으로 벌어져 1등의 확률이 1에 수렴합니다. 즉 greedy(항상 argmax)와 같아집니다.

</details>

<details><summary>Q2. top-p 0.9는 top-k 50과 무엇이 다른가요?</summary>

top-k는 후보 수가 항상 k개로 고정입니다. top-p는 모델이 확신할 때(한 토큰이 0.95)는 후보가 1개, 애매할 때는 수십 개로 **상황에 따라 후보 수가 변합니다.**

</details>

<details><summary>Q3. 퓨샷 프롬프트와 파인튜닝은 각각 언제 쓰나요?</summary>

예시 몇 개로 해결되고 요청량이 많지 않으면 퓨샷이 빠르고 쌉니다(학습 불필요, 즉시 수정 가능). 프롬프트가 너무 길어지거나, 일관된 형식·말투가 대량으로 필요하거나, 프롬프트로는 도저히 안 되는 행동이면 파인튜닝을 고려합니다.

</details>

## 직접 고쳐보기

1. `sample`에 `top_k=1`을 주면 무엇과 같아지나요? 확인해 보세요.
2. 같은 프롬프트로 `temperature=1.0`에서 5번 생성해 결과가 얼마나 다른지 보세요. `top_p=0.5`를 주면 다양성이 어떻게 변하나요?
3. 퓨샷 예시를 0개(시스템 프롬프트만), 2개, 6개로 바꿔 분류 결과의 형식이 얼마나 잘 지켜지는지 비교하세요.
4. 시스템 프롬프트를 바꿔 "항상 JSON으로 답하는" 챗봇을 만들어 보세요. 0.5B 모델이 얼마나 잘 지키나요?
5. (도전) `greedy`에 KV 캐시를 직접 구현해 보세요: 첫 호출에서 `out = model(ids, use_cache=True)`로 `out.past_key_values`를 받고, 다음부터는 `model(next_id.view(1, 1), past_key_values=past, use_cache=True)`처럼 새 토큰 하나만 넣습니다.
6. (도전) 모델을 `Qwen/Qwen2.5-1.5B-Instruct`로 바꿔 한계 확인 질문의 답이 어떻게 달라지는지 보세요.
