# 진짜 VLM 파인튜닝

> ⏱ 60분 · T4 GPU 필요 (약 10분)

**목표:** 제대로 사전학습된 VLM을 불러와 써 보고, LoRA로 **내가 원하는 출력 형식**에 맞게 파인튜닝합니다. [LLM 파인튜닝](#33-llm-finetune) 레슨(LoRA)와 [미니 VLM](#41-mini-vlm) 레슨(VLM 구조)의 합체입니다.

## 모델 불러오기: SmolVLM-256M

파라미터 2.5억 개의 아주 작은 VLM입니다. 작아도 구조는 큰 모델들과 같습니다.

```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
name = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(name)   # 토크나이저 + 이미지 전처리기를 묶은 것
processor.image_processor.do_image_splitting = False   # 이미지를 여러 조각으로 나누지 않음 → 이미지당 토큰 64개로 고정 (빠름)
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32).to(device)

print([n for n, _ in model.model.named_children()])   # ['vision_model', 'connector', 'text_model']
```

`vision_model`(눈), `connector`(통역사), `text_model`(뇌). [미니 VLM](#41-mini-vlm) 레슨에서 직접 만든 세 부품 그대로입니다.

## 사용해 보기: 설명도 하고, 질문에도 답합니다

```python
ds = load_dataset("jxie/flickr8k", split="validation")   # 1000장
train_ds, test_ds = ds.select(range(800)), ds.select(range(800, 1000))

def make_prompt(question):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)

@torch.no_grad()
def ask(model, image, question, max_new_tokens=60):
    inputs = processor(text=make_prompt(question), images=[image.convert("RGB")], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(repr(make_prompt("Describe this image.")))
image = test_ds[0]["image"]
for q in ["Describe this image.", "How many people are in the image?", "What colors stand out?"]:
    print(f"Q: {q}\nA: {ask(model, image, q)}\n")
image
```

## 파인튜닝 과제: "장황한 설명" → "한 문장 캡션"

기본 모델은 설명을 길게 늘어놓습니다. 검색용 캡션처럼 **짧은 한 문장**으로 답하게 만들고 싶다고 합시다. 프롬프트로 부탁할 수도 있지만, 파인튜닝하면 모델의 기본 행동 자체가 바뀝니다. 여러분의 실제 과제(불량품 판정, 영수증 항목 추출, 의료 영상 소견 등)도 데이터만 다를 뿐 같은 절차입니다.

```python
QUESTION = "Describe this image."
for i in range(3):
    print("기본 모델:", ask(model, test_ds[i]["image"], QUESTION))
    print("원하는 답:", test_ds[i]["caption_0"], "\n")
```

## LoRA 붙이기 — LLM 부분에만

`q_proj` 같은 이름은 vision_model에도 있으므로, 정규식으로 **text_model 안의 것만** 고릅니다. 어디를 학습시킬지 고르는 것이 VLM 파인튜닝의 핵심 결정입니다.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                    target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj)")
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

## 배치 만들기

[LLM 파인튜닝](#33-llm-finetune) 레슨과 같습니다. `프롬프트 + 정답`을 통째로 넣고, 프롬프트(이미지 토큰 포함) 부분은 라벨을 `-100`으로 가립니다. 이미지 분할을 꺼 두었기 때문에 프롬프트 길이가 모든 샘플에서 같아 마스킹이 간단합니다.

```python
prompt = make_prompt(QUESTION)
end = "<end_of_utterance>"   # 이 모델이 "답변 끝"을 표시하는 토큰
processor.tokenizer.padding_side = "right"
prompt_len = processor(text=prompt, images=[train_ds[0]["image"].convert("RGB")], return_tensors="pt")["input_ids"].shape[1]

def collate(rows):
    texts = [f"{prompt} {r['caption_0']}{end}" for r in rows]
    images = [[r["image"].convert("RGB")] for r in rows]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[:, :prompt_len] = -100                    # 프롬프트 + 이미지 토큰
    labels[batch["attention_mask"] == 0] = -100      # 패딩
    batch["labels"] = labels
    return batch

b = collate([train_ds[0], train_ds[1]])
print({k: tuple(v.shape) for k, v in b.items()})
print("손실을 계산하는 부분:", processor.decode(b["labels"][0][b["labels"][0] != -100]))
```

## 학습

```python
from torch.utils.data import DataLoader

dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

model.train()
for step, batch in enumerate(dl):          # 1 에폭 = 200 스텝
    batch = {k: v.to(device) for k, v in batch.items()}
    loss = model(**batch).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")
```

## 학습 후

```python
model.eval()
for i in range(3):
    print("파인튜닝 후:", ask(model, test_ds[i]["image"], QUESTION))
    print("정답 예시  :", test_ds[i]["caption_0"], "\n")

print(ask(model, test_ds[0]["image"], "How many people are in the image?"))   # 다른 질문에는 어떻게 답할까요?
model.save_pretrained("caption-lora")
```

## 여기까지 왔다면

직선 하나를 맞추던 네 줄짜리 루프로 LLM과 VLM까지 학습시켰습니다. 이제 남은 것은 규모와 데이터입니다. 다음으로 해볼 만한 것들:

- **내 데이터셋 만들기:** `(이미지, 질문, 답)` 50~200개만 있어도 좁은 과제는 눈에 띄게 좋아집니다. `datasets.Dataset.from_list([...])`로 만들 수 있습니다.
- **더 큰 모델:** `SmolVLM-500M`, `SmolVLM2-2.2B`, `Qwen2.5-VL-3B`. 메모리가 부족하면 4bit 양자화 + LoRA(QLoRA)를 찾아보세요.
- **학습 도구:** 직접 짠 루프 대신 Hugging Face `Trainer`나 `trl`의 `SFTTrainer`를 쓰면 체크포인트, 로깅, 혼합 정밀도, 그래디언트 누적을 알아서 해 줍니다. 안에서 하는 일은 여러분이 이미 아는 그 루프입니다.
- **평가:** 눈으로 보는 것을 넘어, 테스트셋에 대해 정량 지표를 만들어 보세요. "좋아졌다"를 숫자로 말할 수 있어야 실험이 됩니다.

## 핵심 정리

- 실제 VLM도 vision_model / connector / text_model 세 부분으로 이루어져 있습니다.
- `processor`가 텍스트 토큰화와 이미지 전처리를 함께 처리하고, 채팅 템플릿에 이미지 자리를 넣어 줍니다.
- 어느 부분에 LoRA를 붙일지(`target_modules`)가 VLM 파인튜닝의 핵심 결정입니다.
- 학습 절차는 LLM 파인튜닝과 같습니다: 프롬프트+정답을 넣고 정답 부분에서만 손실을 계산합니다.
- 한 가지 형식만 학습시키면 다른 능력이 약해질 수 있습니다(catastrophic forgetting). 항상 학습하지 않은 과제도 점검하세요.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 프롬프트로 '짧게 답해'라고 시키는 것과 파인튜닝의 차이는?</summary>

프롬프트는 매번 지시해야 하고 모델이 안 따를 수도 있습니다. 파인튜닝은 모델의 기본 행동 자체를 바꾸므로 지시 없이도 일관되게 그 형식이 나옵니다. 대신 데이터와 학습이 필요하고 다른 능력이 약해질 위험이 있습니다.

</details>

<details><summary>Q2. <code>target_modules</code>를 정규식으로 <code>text_model</code>에 한정한 이유는?</summary>

`q_proj` 같은 이름은 vision_model 안에도 있어서, 이름만 쓰면 눈과 뇌 양쪽에 LoRA가 붙습니다. 이 과제는 '보는 능력'이 아니라 '말하는 형식'을 바꾸는 것이므로 LLM 쪽만 학습합니다.

</details>

<details><summary>Q3. 파인튜닝 후 반드시 확인해야 할 것 두 가지는?</summary>

① 학습에 쓰지 않은 테스트 데이터에서 원하는 행동이 나오는지, ② 학습시키지 않은 다른 능력(다른 질문에 대한 답)이 망가지지 않았는지.

</details>

## 직접 고쳐보기

1. 마지막 셀에서 "How many people…" 질문의 답이 파인튜닝 전과 어떻게 달라졌나요? 한 가지 형식만 학습시키면 다른 능력이 약해지는 현상(**catastrophic forgetting**)을 관찰해 보세요. 어떻게 줄일 수 있을까요? (힌트: 데이터에 다양한 질문 섞기, 학습 스텝 줄이기, `r` 낮추기)
2. `target_modules`의 `text_model`을 `vision_model`로 바꿔 **눈만** 학습시켜 보세요 (vision 쪽 출력 projection 이름은 `out_proj`입니다). 결과가 어떻게 다른가요?
3. `do_image_splitting = True`로 켜면 이미지가 여러 조각으로 나뉘어 토큰 수가 크게 늘어납니다. 이 경우 `prompt_len`이 샘플마다 달라지므로 `collate`를 어떻게 고쳐야 할까요?
4. **내 과제로 바꾸기:** `QUESTION`과 정답을 바꿔 보세요. 예: 질문을 `"Is there a dog in this image? Answer yes or no."`로 하고, 정답은 캡션에 "dog"가 들어있는지로 자동 생성.
