# 6. 나만의 미니 VLM — 눈과 뇌를 이어붙이기

**목표:** 이미지 인코더(눈)와 LLM(뇌)을 **직접 연결**해서, 사진을 보고 설명하는 모델을 만듭니다. LLaVA 같은 실제 VLM의 1단계 학습과 같은 방식입니다.

> Colab에서 **T4 GPU**를 켜세요. 데이터 다운로드(약 1GB)와 학습을 합쳐 15분 정도 걸립니다.

## VLM의 구조는 의외로 단순합니다

```
사진 → [이미지 인코더] → 패치 벡터 49개 (768차원)
                              ↓
                        [프로젝터]  ← 우리가 학습시킬 유일한 부분
                              ↓
                       "가짜 토큰" 49개 (896차원)
                              ↓
        [ 텍스트 토큰 임베딩 … 이미지 토큰 49개 … 텍스트 토큰 임베딩 ] → [LLM] → 설명 문장
```

LLM은 원래 토큰 번호를 **임베딩 벡터**로 바꾼 뒤 처리합니다(레슨 4의 `tok_emb`). 그렇다면 이미지를 **같은 크기의 벡터**로 바꿔서 그 자리에 끼워 넣으면, LLM은 그것을 "처음 보는 단어들"처럼 읽을 수 있습니다. 프로젝터의 임무는 이미지 인코더의 언어를 LLM의 언어로 **통역**하는 것입니다.

- 이미지 인코더: CLIP ViT-B/32 — **얼림**
- LLM: Qwen2.5-0.5B-Instruct — **얼림**
- 프로젝터: 작은 MLP — **이것만 학습** (약 150만 파라미터)

## 데이터: Flickr8k (사진 + 영어 설명 5개씩)

```python
import random
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, CLIPImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)

ds = load_dataset("jxie/flickr8k", split="train")        # 6000장
test_ds = load_dataset("jxie/flickr8k", split="test[:8]")
captions = [[row[f"caption_{k}"] for k in range(5)] for row in ds.remove_columns("image")]
print(len(ds), "장 /", captions[0])
ds[0]["image"]
```

## 눈: 이미지 인코더

ViT는 이미지를 32×32 픽셀 패치로 잘라(224×224 → 7×7=49개) 각 패치를 토큰처럼 다루는 트랜스포머입니다. 레슨 4의 GPT와 거의 같은 구조에서 마스크만 없다고 보면 됩니다.

```python
vit_name = "openai/clip-vit-base-patch32"
processor = CLIPImageProcessor.from_pretrained(vit_name)   # 리사이즈 + 정규화
vit = CLIPVisionModel.from_pretrained(vit_name).to(device).eval()

@torch.no_grad()
def image_features(images):
    pixels = processor(images=[im.convert("RGB") for im in images], return_tensors="pt").pixel_values.to(device)
    return vit(pixel_values=pixels).last_hidden_state[:, 1:]   # 맨 앞 요약(CLS) 토큰 제외 → [B, 49, 768]

print(image_features([ds[0]["image"]]).shape)
```

인코더는 얼려둘 것이므로 **같은 사진의 특징은 항상 같습니다.** 미리 한 번만 계산해 두면 학습이 훨씬 빨라집니다.

```python
feats = torch.cat([image_features(ds[i:i + 64]["image"]).half().cpu() for i in range(0, len(ds), 64)])
print(feats.shape)   # [6000, 49, 768]
```

## 뇌: LLM (얼림)

```python
llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float32).to(device).eval()
for p in llm.parameters():
    p.requires_grad = False
embed = llm.get_input_embeddings()    # 토큰 번호 → 벡터. 레슨 4의 tok_emb
print(embed)
```

## 통역사: 프로젝터

```python
projector = nn.Sequential(
    nn.Linear(vit.config.hidden_size, llm.config.hidden_size),   # 768 → 896
    nn.GELU(),
    nn.Linear(llm.config.hidden_size, llm.config.hidden_size),
).to(device)
print("학습할 파라미터:", sum(p.numel() for p in projector.parameters()))
```

## 입력 조립: 텍스트 임베딩 사이에 이미지 끼워 넣기

채팅 형식의 프롬프트에서 `<image>` 자리를 기준으로 앞뒤를 나누고, 그 사이에 이미지 토큰을 넣습니다. 이번엔 토큰 번호(`input_ids`) 대신 **임베딩을 직접**(`inputs_embeds`) LLM에 넣는다는 점이 레슨 5와 다릅니다.

```python
prompt = tok.apply_chat_template([{"role": "user", "content": "<image>\nDescribe this image."}],
                                 tokenize=False, add_generation_prompt=True)
pre, post = prompt.split("<image>")
pre_ids = tok(pre, add_special_tokens=False).input_ids
post_ids = tok(post, add_special_tokens=False).input_ids

def build_batch(img_feats, caps):
    seqs = [tok(c + tok.eos_token, add_special_tokens=False).input_ids for c in caps]
    n = max(len(s) for s in seqs)
    ids = lambda rows: torch.tensor(rows, device=device)
    img = projector(img_feats.to(device).float())                                  # [B, 49, 896]
    B, P = img.shape[:2]
    pre_e = embed(ids([pre_ids] * B))                                              # [B, 앞, 896]
    txt_e = embed(ids([post_ids + s + [tok.pad_token_id] * (n - len(s)) for s in seqs]))
    embeds = torch.cat([pre_e, img, txt_e], dim=1)
    ignore = [-100] * (len(pre_ids) + P + len(post_ids))                           # 프롬프트·이미지 위치는 손실 제외
    labels = ids([ignore + s + [-100] * (n - len(s)) for s in seqs])               # 설명 문장에서만 손실 계산
    mask = ids([[1] * (len(ignore) + len(s)) + [0] * (n - len(s)) for s in seqs])
    return embeds, mask, labels

e, m, l = build_batch(feats[:2], [captions[0][0], captions[1][0]])
print(e.shape, m.shape, l.shape)
```

## 학습 전: LLM은 이미지 토큰을 전혀 이해하지 못합니다

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def describe(image, max_new_tokens=30):
    img = projector(image_features([image]).float())
    embeds = torch.cat([embed(torch.tensor([pre_ids], device=device)), img,
                        embed(torch.tensor([post_ids], device=device))], dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
    out = llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def show(n=4):
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, row in zip(axes, test_ds):
        ax.imshow(row["image"]); ax.axis("off")
        ax.set_title(describe(row["image"]), fontsize=9, wrap=True)
    plt.show()

show()
```

## 학습 — 또 그 루프

기울기는 얼어 있는 LLM을 **통과해서** 프로젝터까지 흘러갑니다. LLM의 가중치는 변하지 않지만, "LLM이 알아들으려면 이미지 벡터가 어떤 모양이어야 하는지"를 프로젝터에 알려주는 통로 역할을 합니다.

```python
epochs, batch_size = 2, 16
optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)

for epoch in range(epochs):
    perm = torch.randperm(len(feats))
    for step, i in enumerate(range(0, len(perm), batch_size)):
        idx = perm[i:i + batch_size]
        caps = [random.choice(captions[j]) for j in idx.tolist()]    # 5개 설명 중 하나를 무작위로
        embeds, mask, labels = build_batch(feats[idx], caps)
        loss = llm(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step:4d}  loss {loss.item():.4f}")
```

## 학습 후

```python
show()
torch.save(projector.state_dict(), "projector.pt")   # 저장할 것은 프로젝터뿐 (약 6MB)
```

완벽하진 않습니다. 개·눈·잔디처럼 데이터에 자주 나온 장면은 잘 맞히지만, 사람 사진에는 "A man in a black shirt…" 같은 흔한 문장을 반복하기도 합니다(6000장은 아주 적은 데이터입니다). 그래도 학습 전의 "이미지를 볼 수 없습니다"와 비교해 보세요. **LLM도 이미지 인코더도 1바이트도 바꾸지 않았는데**, 작은 통역사 하나만 학습시켜서 LLM이 사진을 "읽게" 된 것입니다. 실제 VLM(LLaVA 등)은 이 1단계를 수십만 장으로 학습한 뒤, 2단계에서 LLM까지 함께 파인튜닝해 질문 답변 능력을 키웁니다.

## 직접 고쳐보기

1. 프로젝터를 `nn.Linear(768, 896)` 한 층으로 줄여보세요. loss와 설명 품질이 어떻게 달라지나요?
2. `image_features`에서 `[:, 1:]`을 `[:, :1]`로 바꿔 CLS 토큰 **1개만** 쓰도록 해보세요 (`feats`를 다시 계산해야 합니다). 이미지 토큰 49개 vs 1개의 차이는?
3. 직접 찍은 사진으로 시험해 보세요: Colab 왼쪽 파일 탭에 업로드 후 `from PIL import Image; describe(Image.open("내사진.jpg"))`. Flickr8k에 없는 종류의 사진(음식, 문서, 실내)에서는 어떻게 되나요? 왜일까요?
4. 프롬프트의 `"Describe this image."`를 바꿔서 학습/추론해 보세요.
5. (도전) 2단계 학습: 레슨 5의 LoRA를 `llm`에 붙이고, 프로젝터와 LoRA를 **함께** 학습시켜 보세요. 옵티마이저에 두 파라미터 그룹을 모두 넣어야 합니다.
6. (도전) 설명을 한국어로: `captions`를 번역(예: 레슨 5의 모델에게 시키기)해서 한국어 캡션 모델을 만들어 보세요.
