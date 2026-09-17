# 토크나이저 — LLM이 글을 보는 방식

> ⏱ 45분 · CPU로 충분

**목표:** LLM은 글자를 보지 않습니다. **토큰 번호**를 봅니다. 실제 LLM이 쓰는 BPE 토크나이저를 30줄로 직접 만들어 보고, 토큰화가 비용·성능·모델의 이상한 행동에 어떤 영향을 주는지 이해합니다.

## 글을 숫자로 바꾸는 세 가지 방법

| 방식 | 예: "unbelievable" | 장점 | 단점 |
|---|---|---|---|
| 글자 단위 | u, n, b, e, l, … (12개) | 어휘가 작음, 모르는 단어 없음 | 시퀀스가 너무 길어짐 |
| 단어 단위 | unbelievable (1개) | 짧음 | 어휘가 수백만 개, 처음 보는 단어는 처리 불가 |
| **서브워드** | un, believ, able (3개) | 둘의 절충 | 학습이 필요 |

현재 LLM은 전부 서브워드 방식이고, 그중 대표가 **BPE (Byte Pair Encoding)** 입니다. 아이디어는 단순합니다: **가장 자주 붙어 나오는 쌍을 하나로 합치기**를 반복한다.

## 출발점: 모든 글은 바이트다

어떤 언어든 UTF-8로 바꾸면 0~255 사이 숫자(바이트)의 나열이 됩니다. 여기서 시작하면 "모르는 글자"가 원천적으로 없습니다.

```python
for s in ["hello", "안녕", "🙂"]:
    b = list(s.encode("utf-8"))
    print(f"{s!r}: {len(s)}글자 → {len(b)}바이트 {b}")
```

영어는 글자당 1바이트, **한글은 글자당 3바이트**입니다. 이 차이를 기억해 두세요.

## BPE 학습 직접 구현하기

```python
import urllib.request
from collections import Counter

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")[:50000]

def count_pairs(ids):
    return Counter(zip(ids, ids[1:]))            # 이웃한 쌍의 등장 횟수

def merge(ids, pair, new_id):                    # ids 안의 모든 pair를 new_id 하나로 교체
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train_bpe(text, num_merges):
    ids = list(text.encode("utf-8"))
    merges = {}                                  # (a, b) → 새 토큰 번호. 이 표가 곧 "학습된 토크나이저"
    for k in range(num_merges):
        pair = count_pairs(ids).most_common(1)[0][0]
        merges[pair] = 256 + k                   # 0~255는 바이트, 그 뒤부터 새 토큰
        ids = merge(ids, pair, 256 + k)
    return merges

merges = train_bpe(text, 200)
print("어휘 크기:", 256 + len(merges))
```

무엇이 합쳐졌는지 봅시다.

```python
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():            # 합쳐진 순서대로 풀어 쓰면 각 토큰의 실제 문자열
    vocab[new_id] = vocab[a] + vocab[b]

print("처음 합쳐진 것들:", [vocab[256 + k].decode() for k in range(15)])
print("나중에 합쳐진 것들:", [vocab[256 + k].decode() for k in range(185, 200)])
```

처음에는 `e␣`, `th` 같은 글자 쌍이, 나중에는 `know`, `but␣`, `will␣` 같은 **단어 조각**이 토큰이 됩니다(␣는 공백). 누구도 영어 문법을 알려주지 않았습니다. 빈도만 셌을 뿐입니다.

## 인코딩과 디코딩

```python
def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) > 1:
        # 현재 쌍들 중 "가장 먼저 학습된 병합"부터 적용
        pair = min(count_pairs(ids), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

s = "What is the meaning of this?"
ids = encode(s)
print(len(s), "글자 →", len(ids), "토큰")
print([vocab[i].decode() for i in ids])
assert decode(ids) == s
```

이제 이 토크나이저에 한국어를 넣어 봅시다.

```python
s = "토크나이저를 직접 만들었다"
ids = encode(s)
print(len(s), "글자 →", len(ids), "토큰")    # 글자 수보다 토큰이 훨씬 많습니다
assert decode(ids) == s                       # 그래도 정보 손실은 없습니다
```

영어로만 학습한 토크나이저는 한글 병합 규칙을 하나도 배우지 못해, 한글을 **바이트 단위로 낱낱이** 쪼갭니다. 같은 내용인데 토큰이 3배 이상 드는 셈입니다.

## 실제 LLM의 토크나이저와 비교

실제 토크나이저도 원리는 같습니다. 병합을 200번이 아니라 수만~수십만 번 했을 뿐입니다.

```python
from transformers import AutoTokenizer

gpt2 = AutoTokenizer.from_pretrained("gpt2")                        # 2019년, 거의 영어로만 학습
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # 다국어로 학습
print("어휘 크기:", len(gpt2), len(qwen))

samples = ["Deep learning is fun to learn by building things.",
           "딥러닝은 직접 만들어 보면서 배우는 것이 가장 재미있습니다."]
for s in samples:
    print(f"\n{s}\n  GPT-2: {len(gpt2(s).input_ids):3d} 토큰 | Qwen2.5: {len(qwen(s).input_ids):3d} 토큰")
print([qwen.decode(i) for i in qwen(samples[1]).input_ids])
```

출력에 `�`가 보이나요? 한글 한 글자(3바이트)가 두 토큰에 걸쳐 쪼개진 것입니다. 토큰 하나만 떼어 보면 온전한 글자가 아니어서 깨져 보이지만, 이어 붙이면 정상적으로 복원됩니다.

**토큰 수는 곧 돈이고 속도이고 기억력입니다.** API 요금은 토큰당 매겨지고, 생성 속도는 초당 토큰으로 재며, 컨텍스트 길이 한도도 토큰 기준입니다. 한국어를 잘 지원하는 모델을 고를 때 "한국어 토큰 효율"을 보는 이유입니다.

## 토큰화 때문에 생기는 LLM의 이상한 행동

```python
for w in ["strawberry", " strawberry", "Strawberry", "12345678", "3.14159"]:
    print(f"{w!r:15} Qwen: {[qwen.decode(i) for i in qwen(w).input_ids]}  GPT-2: {[gpt2.decode(i) for i in gpt2(w).input_ids]}")
```

- **철자 세기를 못한다:** 모델은 `strawberry`를 글자 10개가 아니라 토큰 몇 개로 봅니다. "r이 몇 개?"가 어려운 이유입니다.
- **공백·대소문자에 민감하다:** `"strawberry"`와 `" strawberry"`는 서로 다른 토큰입니다.
- **숫자 계산이 약하다:** GPT-2는 긴 숫자를 임의의 조각으로 나눕니다. 자릿수가 뒤섞여 산수가 어려워집니다. 그래서 Qwen 같은 최근 토크나이저는 숫자를 **한 자리씩** 쪼개도록 설계되어 있습니다.

## 특수 토큰

일반 텍스트에는 나오지 않고 **역할**을 표시하는 토큰들입니다. 대화의 시작·끝, 문서의 끝, 이미지가 들어갈 자리 등을 나타냅니다. 뒤의 파인튜닝 레슨에서 계속 등장합니다.

```python
print(qwen.special_tokens_map)
chat = qwen.apply_chat_template([{"role": "user", "content": "안녕?"}], tokenize=False, add_generation_prompt=True)
print(chat)   # <|im_start|>, <|im_end|> 가 대화의 구조를 표시하는 특수 토큰
```

## 핵심 정리

- LLM의 입력과 출력은 글자가 아니라 **토큰 번호**입니다. 토크나이저와 모델은 항상 짝입니다.
- BPE: 바이트에서 시작해 가장 흔한 쌍을 반복해 합칩니다. 병합 규칙 표가 곧 토크나이저입니다.
- 토크나이저도 **데이터로 학습**됩니다. 학습 데이터에 적었던 언어는 토큰이 많이 듭니다.
- 토큰 수 = 비용, 속도, 컨텍스트 소모량.
- 철자·숫자·공백 관련 실수는 모델의 지능보다 토큰화 탓인 경우가 많습니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 바이트 단위에서 시작하는 BPE에는 왜 "모르는 단어(unknown token)"가 없나요?</summary>

어떤 문자열이든 UTF-8 바이트(0~255)의 나열로 표현할 수 있고, 그 256개가 기본 어휘에 들어 있기 때문입니다. 병합 규칙이 없으면 바이트 단위로 길게 표현될 뿐입니다.

</details>

<details><summary>Q2. 어휘 크기를 키우면(병합을 더 많이 하면) 무엇이 좋아지고 무엇이 나빠지나요?</summary>

같은 글이 더 적은 토큰으로 표현되어 빠르고 더 긴 글을 볼 수 있습니다. 대신 임베딩 표와 출력층(어휘 크기 × 임베딩 차원)이 커지고, 드물게 나오는 토큰은 충분히 학습되지 못합니다.

</details>

<details><summary>Q3. 학습된 LLM에 다른 모델의 토크나이저를 쓰면 어떻게 되나요?</summary>

같은 번호가 전혀 다른 조각을 뜻하게 되어 모델은 의미 없는 입력을 받습니다. 토크나이저는 모델 가중치와 함께 배포되며 바꿔 쓸 수 없습니다.

</details>

## 직접 고쳐보기

1. `num_merges`를 50, 200, 1000으로 바꿔 같은 영어 문장의 토큰 수를 비교하세요(1000은 1~2분 걸립니다).
2. 학습 텍스트를 한국어로 바꿔 보세요(직접 쓴 글이나 뉴스 기사 몇 편을 `text = """..."""`에 붙여넣기). 어떤 조각이 토큰이 되나요? 조사(`은`, `는`, `을`)나 어미(`습니다`)가 나오나요?
3. 자주 쓰는 한국어 문장 5개로 GPT-2와 Qwen의 평균 "글자당 토큰 수"를 계산해 보세요.
4. (도전) `encode`는 매번 전체를 다시 세기 때문에 느립니다. 더 빠르게 만들 방법을 생각해 보세요. 실제 구현(`tiktoken`, `tokenizers`)은 Rust로 작성되어 있습니다.
