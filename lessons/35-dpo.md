# 선호 학습 (DPO) — "이 답이 저 답보다 낫다"로 학습하기

> ⏱ 70분 · T4 GPU 필요 (약 5분)

**목표:** ChatGPT 같은 모델을 만드는 마지막 단계인 **선호 학습**을 직접 구현합니다. 정답을 알려주는 대신 "두 답 중 어느 쪽이 나은지"만 알려줘서 모델의 행동을 바꿉니다. 라이브러리 없이 DPO 손실을 10줄로 짭니다.

## LLM이 만들어지는 세 단계

| 단계 | 데이터 | 배우는 것 | 이 코스에서 |
|---|---|---|---|
| ① 사전학습 | 방대한 텍스트 | 언어와 지식 (다음 토큰 예측) | [미니 GPT](#31-mini-gpt) |
| ② SFT (지도 파인튜닝) | (질문, 모범 답) | 지시를 따르는 형식 | [LLM 파인튜닝](#33-llm-finetune) |
| ③ **선호 학습** | (질문, 좋은 답, 나쁜 답) | 사람이 **더 좋아하는** 답 | 이 레슨 |

SFT의 한계: "좋은 답"을 직접 써 줘야 하고, **무엇이 나쁜지**는 가르칠 수 없습니다. 반면 사람은 답을 쓰는 것보다 **둘 중 고르는 것**을 훨씬 쉽게, 일관되게 합니다. 선호 학습은 이 비교 데이터를 씁니다.

- **RLHF:** 선호 데이터로 "보상 모델"을 먼저 학습시키고, 강화학습(PPO)으로 LLM이 높은 보상을 받도록 학습. 강력하지만 복잡합니다.
- **DPO:** 보상 모델과 강화학습 없이 선호 데이터에서 **바로** LLM을 학습. 간단해서 널리 쓰입니다.

## 준비: 모델과 LoRA

과제는 **"장황하게 답하는 모델을 간결하게 답하도록"** 만드는 것입니다.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

def to_prompt(question, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def chat(question, system=None, max_new_tokens=150):
    model.eval()
    inputs = tok(to_prompt(question, system), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

test_questions = ["지구는 왜 둥글어?", "커피를 마시면 왜 잠이 깨?", "비행기는 어떻게 하늘을 날아?", "김치는 왜 시어져?"]

def report():
    lengths = []
    for q in test_questions:
        a = chat(q); lengths.append(len(tok(a).input_ids))
        print(f"Q: {q}\nA ({lengths[-1]} 토큰): {a[:150].replace(chr(10), ' ')}{'…' if len(a) > 150 else ''}\n")
    print("평균 답변 길이:", sum(lengths) / len(lengths), "토큰")

report()
```

## 선호 데이터 만들기

각 질문마다 **chosen(선호하는 답)** 과 **rejected(선호하지 않는 답)** 가 필요합니다. 실무에서는 모델이 내놓은 답 여러 개를 사람이 비교해서 만듭니다. 여기서는 사람 대신 간단한 요령을 씁니다. 같은 모델에게 **짧게 답하라는 시스템 프롬프트를 줬을 때의 답을 chosen**, 아무 지시 없이 평소대로 내놓은 **장황한 답을 rejected**로 삼습니다. 학습이 끝나면 시스템 프롬프트 없이도 짧게 답하는 모델이 됩니다. 프롬프트로 끌어낸 행동을 가중치에 새겨 넣는 셈입니다.

```python
questions = ["하늘은 왜 파란가요?", "물은 몇 도에서 끓어?", "광합성이 뭐야?", "달은 왜 모양이 바뀌어?", "무지개는 어떻게 생겨?",
             "왜 겨울에 더 추워?", "번개는 왜 쳐?", "사람은 왜 잠을 자야 해?", "바닷물은 왜 짜?", "인터넷은 어떻게 작동해?",
             "백신은 어떻게 효과를 내?", "얼음은 왜 물에 떠?", "지진은 왜 일어나?", "나뭇잎은 왜 가을에 색이 변해?", "전기는 어떻게 만들어져?",
             "혈액은 왜 빨개?", "별은 왜 반짝여?", "소리는 어떻게 전달돼?", "운동하면 왜 땀이 나?", "밀물과 썰물은 왜 생겨?"]
SHORT = "질문에 한 문장으로만 아주 짧게 답한다. 목록이나 부연 설명을 붙이지 않는다."

pairs = [{"prompt": to_prompt(q),                      # 학습할 때의 프롬프트에는 시스템 프롬프트가 없다
          "chosen": chat(q, system=SHORT),             # 짧게 답하라고 시켰을 때의 답
          "rejected": chat(q)} for q in questions]     # 평소의 장황한 답
print(pairs[0]["chosen"], "\n---\n", pairs[0]["rejected"][:200], "…")
```

## DPO의 아이디어

모델이 어떤 답 `y`를 생성할 **로그 확률** `log π(y|x)`는 각 토큰의 로그 확률을 더한 것입니다. DPO가 원하는 것은 단순합니다.

> **chosen의 확률은 올리고, rejected의 확률은 내려라. 단, 원래 모델(reference)에서 너무 멀어지지는 마라.**

```
chosen 개선폭   = log π(chosen)   − log π_ref(chosen)      ← 원래 모델 대비 얼마나 더 선호하게 됐나
rejected 개선폭 = log π(rejected) − log π_ref(rejected)
loss = −log sigmoid( β × (chosen 개선폭 − rejected 개선폭) )
```

β는 "원래 모델에서 벗어나는 것을 얼마나 허용할지"를 조절합니다. reference가 기준점 역할을 하기 때문에 모델이 rejected를 피하려다 언어 능력 자체를 망가뜨리는 것을 막아 줍니다.

```python
def sequence_logprob(model, prompt, answer):
    """모델이 prompt 뒤에 answer를 생성할 로그 확률 (answer 토큰들의 로그 확률 합)"""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + a_ids], device=device)
    logits = model(ids).logits[0, :-1]                    # 위치 t의 출력은 t+1번째 토큰에 대한 예측
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(1, ids[0, 1:, None])[:, 0]   # 실제로 나온 토큰의 로그 확률만 뽑기
    return token_logp[len(p_ids) - 1:].sum()              # answer 부분만 합산

def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return -F.logsigmoid(margin), margin
```

reference 모델은 따로 불러올 필요가 없습니다. **LoRA 어댑터를 잠시 끄면** 그것이 원래 모델입니다. reference는 변하지 않으므로 미리 한 번만 계산해 둡니다.

```python
model.eval()
with torch.no_grad(), model.disable_adapter():
    for p in pairs:
        p["ref_c"] = sequence_logprob(model, p["prompt"], p["chosen"])
        p["ref_r"] = sequence_logprob(model, p["prompt"], p["rejected"])
print(f"reference가 보는 로그 확률 — chosen: {pairs[0]['ref_c'].item():.1f}, rejected: {pairs[0]['ref_r'].item():.1f}")
```

로그 확률은 토큰마다 음수가 더해지므로 긴 답일수록 값이 작습니다. 중요한 것은 절대값이 아니라, 학습하면서 이 값들이 **reference 대비 어느 쪽으로 움직이는가**입니다.

## 학습

```python
import random

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
accum = 4                                     # 4쌍마다 한 번 갱신 (그래디언트 누적)

model.train()
for epoch in range(3):
    random.shuffle(pairs)
    total, wins = 0, 0
    for i, p in enumerate(pairs):
        pi_c = sequence_logprob(model, p["prompt"], p["chosen"])
        pi_r = sequence_logprob(model, p["prompt"], p["rejected"])
        loss, margin = dpo_loss(pi_c, pi_r, p["ref_c"], p["ref_r"])
        (loss / accum).backward()
        if (i + 1) % accum == 0:
            optimizer.step(); optimizer.zero_grad()
        total += loss.item(); wins += (margin > 0).item()
    print(f"epoch {epoch+1}  loss {total/len(pairs):.4f}  chosen을 더 선호하게 된 비율 {wins/len(pairs):.2f}")
```

초기 loss는 `-log sigmoid(0) = ln 2 ≈ 0.693`입니다(policy와 reference가 같으므로 margin이 0). [학습 잘 시키는 법](#23-training-recipes)에서 배운 "초기 loss 확인"을 여기서도 할 수 있습니다.

## 결과

```python
report()
```

학습에 쓰지 않은 질문에도 시스템 프롬프트 없이 짧게 답합니다(평균 150 → 50 토큰 안팎). SFT처럼 "이렇게 써라"를 보여준 것이 아니라 **"이쪽이 저쪽보다 낫다"** 는 신호만으로 행동이 바뀌었습니다.

두 가지를 꼭 짚고 넘어가세요.

- **내용은 여전히 틀립니다.** 0.5B 모델의 지식 수준은 그대로입니다. 선호 학습은 길이·말투·형식 같은 **행동**을 바꾸지, 모르는 것을 알게 해 주지 않습니다.
- **데이터 품질이 곧 결과입니다.** `pairs`의 chosen을 몇 개 출력해 보세요. 중국어가 섞였거나 틀린 답도 있을 것입니다. 우리는 그것까지 "선호하는 답"이라고 가르친 셈입니다. 실무에서 선호 데이터를 사람이 검수하고 걸러내는 데 가장 많은 비용을 쓰는 이유입니다.

## 핵심 정리

- LLM 학습의 세 단계: 사전학습 → SFT → 선호 학습.
- 선호 데이터는 `(prompt, chosen, rejected)` 세 쌍. 사람은 쓰는 것보다 고르는 것을 잘합니다.
- 문장의 로그 확률 = 토큰 로그 확률의 합. DPO는 chosen과 rejected의 로그 확률 **차이**를 reference 대비 벌립니다.
- reference 모델이 기준점이 되어 모델이 망가지는 것을 막고, β가 그 세기를 조절합니다.
- LoRA를 쓰면 어댑터를 끄는 것만으로 reference 모델을 얻습니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. "간결한 답"을 가르치는 것이 목적이라면 SFT로도 되지 않나요? 선호 학습만의 장점은?</summary>

SFT로도 가능합니다. 선호 학습의 장점은 ① 모범 답을 쓰기 어려워도 비교만 할 수 있으면 되고(예: "더 공손한 답", "더 안전한 답"), ② **하지 말아야 할 것**(rejected)을 명시적으로 가르칠 수 있으며, ③ 모델 자신의 출력을 기준으로 개선한다는 점입니다.

</details>

<details><summary>Q2. DPO loss에서 reference 항을 빼 버리면 어떤 위험이 있나요?</summary>

rejected의 확률을 낮추는 가장 쉬운 방법은 언어 모델로서의 능력을 통째로 망가뜨리는 것입니다. 기준점이 없으면 확률 차이를 무한정 벌리려 하면서 문법이 깨지거나 같은 말만 반복하는 모델이 될 수 있습니다.

</details>

<details><summary>Q3. 학습 첫 스텝의 loss가 0.693이 아니라면 무엇을 의심해야 하나요?</summary>

policy와 reference의 로그 확률이 같지 않다는 뜻입니다. reference 계산 시와 학습 시의 조건 차이(예: dropout이 켜진 train 모드와 eval 모드), 토큰화 방식 불일치, 어댑터가 제대로 꺼지지 않은 경우 등을 의심합니다. LoRA의 B 행렬은 0으로 초기화되므로 처음에는 정확히 같아야 합니다.

</details>

## 직접 고쳐보기

1. `beta`를 0.01과 0.5로 바꿔 보세요. 답변 길이와 품질이 어떻게 달라지나요?
2. 에폭을 10으로 늘리면 어떻게 되나요? 답이 지나치게 짧아지거나 이상해지지 않는지 확인하세요(과최적화).
3. chosen과 rejected를 **서로 바꿔서** 학습하면? "더 장황한 모델"이 만들어지나요?
4. 다른 선호를 가르쳐 보세요: chosen을 "존댓말 답", rejected를 "반말 답"으로. 또는 chosen을 "~입니다. 더 궁금한 점이 있으신가요?"로 끝나는 답으로.
5. (도전) 같은 데이터의 chosen만 가지고 [LLM 파인튜닝](#33-llm-finetune) 방식(SFT)으로 학습한 모델과 결과를 비교해 보세요.
6. (도전) Hugging Face `trl` 라이브러리의 `DPOTrainer`로 같은 학습을 해 보세요. 안에서 하는 일은 이 레슨의 코드와 같습니다.
