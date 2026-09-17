# LLM 파인튜닝 — Hugging Face와 LoRA

> ⏱ 60분 · T4 GPU 필요 (전체 실행 약 5분)

**목표:** 남이 수조 토큰으로 사전학습해 둔 LLM을 내려받아, **내 데이터로 말투를 바꿔** 봅니다. [미니 GPT](#31-mini-gpt) 레슨에서 만든 것과 같은 구조, 같은 학습 루프입니다.

## 모델 불러오기

[Hugging Face Hub](https://huggingface.co/models)는 모델계의 GitHub입니다. 이름만 주면 가중치를 내려받아 줍니다. 여기서는 한국어가 되는 작은 모델 `Qwen2.5-0.5B-Instruct`(파라미터 5억 개)를 씁니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)

print(model)   # 미니 GPT 레슨과 비교해 보세요: embed_tokens, 24개의 블록(self_attn + mlp), norm, lm_head
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
```

## 토크나이저와 채팅 템플릿

글자 단위였던 [미니 GPT](#31-mini-gpt) 레슨과 달리 서브워드 단위로 쪼갭니다. 그리고 Instruct 모델은 대화를 **정해진 형식의 텍스트**로 바꿔서 학습했기 때문에, 쓸 때도 같은 형식(`apply_chat_template`)을 맞춰줘야 합니다.

```python
ids = tok("딥러닝을 직접 학습시켜 봅시다").input_ids
print(ids)
print([tok.decode(i) for i in ids])

messages = [{"role": "user", "content": "안녕?"}]
print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```

```python
def chat(model, question, max_new_tokens=80):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)   # do_sample=False: 매번 같은 답
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)       # 새로 생성된 부분만

test_questions = ["너는 누구야?", "파이썬이 뭐야?", "오늘 기분 어때?"]
for q in test_questions:
    print(f"Q: {q}\nA: {chat(model, q)}\n")
```

`generate`가 하는 일은 [미니 GPT](#31-mini-gpt) 레슨의 `generate`와 똑같습니다. 다음 토큰을 하나 뽑아 이어붙이고 반복합니다.

## LoRA: 전체를 건드리지 않고 조금만 학습하기

5억 개 파라미터를 전부 학습하려면 메모리가 많이 들고, 데이터가 적으면 모델이 망가지기도 쉽습니다. **LoRA**는 원래 가중치 `W`는 얼려두고([CNN](#22-cnn) 레슨의 freeze), 옆에 작은 행렬 두 개 `A`, `B`를 붙여 `W·x + B·A·x`를 계산합니다. 학습되는 건 `A`, `B`뿐입니다.

원리는 이게 전부입니다:

```python
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False                                    # 원래 가중치는 얼림
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)  # [r, in]  작은 행렬
        self.B = nn.Parameter(torch.zeros(base.out_features, r))       # [out, r] 0으로 시작 → 처음엔 원래 모델과 동일
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

layer = LoRALinear(nn.Linear(896, 896))
print("원래:", 896 * 896, " LoRA가 학습하는 양:", layer.A.numel() + layer.B.numel())
```

실제로는 `peft` 라이브러리가 모델 안의 Linear들을 찾아 이렇게 바꿔 줍니다.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])   # 어텐션의 Q,K,V,출력 projection에 부착
model = get_peft_model(model, config)
model.print_trainable_parameters()   # 전체의 0.2% 정도만 학습
```

## 학습 데이터: 모든 문장을 "~냥"으로 끝내는 고양이 챗봇

효과가 눈에 바로 보이도록 말투를 바꿔 봅니다. 데이터는 (질문, 원하는 답) 쌍입니다.

```python
data = [
    ("너는 누구야?", "나는 딥러닝을 공부하는 고양이 챗봇이다냥! 궁금한 건 뭐든 물어보라냥."),
    ("이름이 뭐야?", "내 이름은 딥냥이다냥. 만나서 반갑다냥!"),
    ("안녕!", "안녕하냥! 오늘도 같이 공부하자냥."),
    ("오늘 날씨 어때?", "나는 창밖을 볼 수 없어서 모른다냥. 그래도 햇볕이 좋으면 낮잠 자기 딱이다냥."),
    ("딥러닝이 뭐야?", "딥러닝은 층을 깊게 쌓은 신경망으로 데이터에서 패턴을 배우는 방법이다냥. 생각보다 어렵지 않다냥!"),
    ("GPU가 왜 필요해?", "행렬 곱셈을 한꺼번에 아주 많이 해야 해서 그렇다냥. GPU는 그런 계산을 동시에 잘한다냥."),
    ("학습률이 뭐야?", "한 번에 파라미터를 얼마나 움직일지 정하는 값이다냥. 너무 크면 발산하고 너무 작으면 느리다냥."),
    ("과적합이 뭐야?", "학습 데이터만 달달 외워서 새로운 데이터에는 약해지는 현상이다냥. 조심해야 한다냥."),
    ("좋아하는 음식이 뭐야?", "당연히 참치다냥! 츄르도 좋아한다냥."),
    ("심심해", "그럼 나랑 모델 하나 학습시켜 보자냥. 시간 가는 줄 모를 거다냥!"),
    ("1 더하기 1은?", "2다냥! 이 정도는 쉽다냥."),
    ("잘 자", "잘 자라냥! 내일 또 보자냥."),
    ("고마워", "천만에다냥! 도움이 됐다니 기쁘다냥."),
    ("파이썬 배우기 어려워?", "처음엔 낯설지만 금방 익숙해진다냥. 매일 조금씩 해보라냥."),
    ("트랜스포머가 뭐야?", "어텐션으로 토큰끼리 정보를 주고받는 신경망 구조다냥. 요즘 LLM은 거의 다 이 구조다냥."),
    ("너 사람이야?", "아니다냥, 나는 고양이 말투를 배운 언어모델이다냥."),
    ("공부하기 싫어", "그럴 땐 딱 10분만 해보라냥. 시작이 반이다냥!"),
    ("추천해줄 취미 있어?", "햇볕 드는 창가에서 책 읽기를 추천한다냥. 졸리면 그냥 자도 된다냥."),
    ("LoRA가 뭐야?", "큰 모델은 얼려두고 작은 행렬만 붙여서 학습하는 방법이다냥. 메모리를 아주 아낄 수 있다냥."),
    ("커피 마셔도 돼?", "적당히 마시면 괜찮다냥. 나는 물이 더 좋다냥."),
]
print(len(data), "개")
```

## 토큰화와 라벨 마스킹

모델에게는 `질문 + 답` 전체를 입력으로 주되, **답 부분에서만 손실을 계산**합니다. 라벨이 `-100`인 위치는 손실에서 제외됩니다(PyTorch `cross_entropy`의 기본 약속).

```python
def encode(question, answer):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    answer_ids = tok(answer + tok.eos_token, add_special_tokens=False).input_ids   # eos: "여기서 말을 끝낸다"도 배워야 함
    return {"input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids}

def collate(batch):   # 길이가 다른 샘플들을 패딩해서 하나의 텐서로
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}

ex = encode(*data[0])
print(tok.decode(ex["input_ids"]))
print("손실을 계산하는 부분:", tok.decode([t for t in ex["labels"] if t != -100]))
```

## 학습 — 익숙한 그 루프

`labels`를 같이 넘기면 Hugging Face 모델이 내부에서 "한 칸 밀기"([미니 GPT](#31-mini-gpt) 레슨의 x, y)와 cross entropy를 계산해 `loss`를 돌려줍니다.

```python
from torch.utils.data import DataLoader

dl = DataLoader([encode(q, a) for q, a in data], batch_size=4, shuffle=True, collate_fn=collate)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

epochs = 5
model.train()
for epoch in range(epochs):
    total = 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch+1}  loss {total / len(dl):.4f}")
```

## 학습 후: 같은 질문 + 처음 보는 질문

```python
model.eval()
for q in test_questions + ["머신러닝이랑 딥러닝은 뭐가 달라?", "주말에 뭐 하면 좋을까?"]:
    print(f"Q: {q}\nA: {chat(model, q)}\n")
```

학습 데이터에 없던 질문에도 "냥" 말투로 답합니다. 모델이 답을 외운 것이 아니라 **말투라는 패턴**을 배웠기 때문입니다. 지식은 사전학습에서, 행동 방식은 파인튜닝에서 온다는 것을 직접 확인한 셈입니다.

## 어댑터 저장과 불러오기

LoRA는 추가된 작은 행렬만 저장하므로 파일이 몇 MB에 불과합니다.

```python
model.save_pretrained("cat-lora")

from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
loaded = PeftModel.from_pretrained(base, "cat-lora").eval()
print(chat(loaded, "너는 누구야?"))
# loaded.merge_and_unload() 를 쓰면 LoRA를 원래 가중치에 합쳐 일반 모델로 만들 수 있습니다
```

## 핵심 정리

- Hugging Face에서는 `from_pretrained(이름)` 한 줄로 모델과 토크나이저를 받습니다. 둘은 항상 짝입니다.
- Instruct 모델은 **채팅 템플릿** 형식으로 입력해야 제 실력이 나옵니다. 학습 때와 추론 때 같은 템플릿을 쓰세요.
- LoRA는 원래 가중치를 얼리고 작은 행렬 A, B만 학습합니다(전체의 1% 미만).
- 질문 부분의 라벨을 `-100`으로 가려 **답 부분에서만** 손실을 계산합니다. 끝 토큰(EOS)도 학습시켜야 말을 멈출 줄 압니다.
- 파인튜닝은 지식보다 **말투·형식·행동**을 바꾸는 데 잘 듣습니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. LoRA에서 B를 0으로 초기화하는 이유는?</summary>

`B·A = 0`이 되어 학습 시작 시점의 모델이 원래 모델과 정확히 같아집니다. 멀쩡한 모델에서 출발해 조금씩 바꿔 나가기 위해서입니다.

</details>

<details><summary>Q2. 학습 데이터에 없던 질문에도 '냥' 말투가 나오는 이유는?</summary>

모델이 개별 답을 외운 것이 아니라 '문장을 ~냥으로 끝낸다'는 패턴을 배웠기 때문입니다. 질문에 대한 지식 자체는 사전학습에서 이미 갖고 있습니다.

</details>

<details><summary>Q3. 답 끝에 <code>eos_token</code>을 붙이지 않고 학습하면?</summary>

모델이 '언제 멈추는지'를 배우지 못해, 답을 끝낸 뒤에도 `max_new_tokens`까지 계속 말을 이어갑니다.

</details>

<details><summary>Q4. 20개 데이터로 에폭을 아주 많이 돌리면 어떤 위험이 있나요?</summary>

과적합입니다. 학습 데이터의 답을 그대로 외우고, 다른 질문에도 외운 문장을 내뱉거나 원래 능력이 망가질 수 있습니다.

</details>

## 직접 고쳐보기

1. `epochs`를 1과 15로 바꿔보세요. 1일 때는 말투가 덜 바뀌고, 15일 때는 학습 데이터의 답을 그대로 외워 버리는지(과적합) 확인하세요.
2. `r=1`과 `r=64`로 바꿔 학습 파라미터 수와 결과를 비교하세요.
3. `encode`에서 `[-100] * len(prompt_ids)`를 `prompt_ids`로 바꾸면(질문 부분도 학습) 무엇이 달라질까요?
4. **내 데이터로 바꾸기:** `data`를 원하는 말투/역할(사투리, 존댓말 상담원, 특정 캐릭터)로 20~50개 직접 써서 학습시켜 보세요. 이 레슨의 진짜 과제입니다.
5. (도전) `target_modules`에 MLP 층(`"gate_proj", "up_proj", "down_proj"`)도 추가해 보세요.
6. (도전) 모델을 `Qwen/Qwen2.5-1.5B-Instruct`로 바꿔보세요. T4에서는 `dtype=torch.float16`으로 불러오면 메모리가 절반이지만 학습이 불안정해질 수 있습니다. 직접 확인해 보세요.
