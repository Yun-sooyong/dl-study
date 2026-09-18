# 나만의 미니 VLM — 눈과 뇌를 이어붙이기

> ⏱ 90분 · T4 GPU 필요 (다운로드 1GB + 학습, 약 15분)

**목표:** 이미지 인코더(눈)와 LLM(뇌)을 **직접 연결**해서, 사진을 보고 설명하는 모델을 만듭니다. LLaVA 같은 실제 VLM의 1단계 학습과 같은 방식입니다.

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

LLM은 원래 토큰 번호를 **임베딩 벡터**로 바꾼 뒤 처리합니다([미니 GPT](#31-mini-gpt) 레슨의 `tok_emb`). 그렇다면 이미지를 **같은 크기의 벡터**로 바꿔서 그 자리에 끼워 넣으면, LLM은 그것을 "처음 보는 단어들"처럼 읽을 수 있습니다. 프로젝터의 임무는 이미지 인코더의 언어를 LLM의 언어로 **통역**하는 것입니다.

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

ViT는 이미지를 32×32 픽셀 패치로 잘라(224×224 → 7×7=49개) 각 패치를 토큰처럼 다루는 트랜스포머입니다. [미니 GPT](#31-mini-gpt) 레슨의 GPT와 거의 같은 구조에서 마스크만 없다고 보면 됩니다.

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

**코드 읽기**

- `CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")` — [CLIP](#40-clip)에서 쓴 모델의 **이미지 인코더만** 불러옵니다(`CLIPModel`이 아님). 텍스트 인코더는 필요 없으니 메모리를 아낍니다. 왜 CLIP의 인코더인가: 글과 짝지어 학습되었기 때문에 그 특징이 "언어로 설명하기 좋은" 정보를 담고 있습니다. LLaVA 등 실제 VLM이 CLIP/SigLIP을 눈으로 쓰는 이유입니다.
- `CLIPImageProcessor` — 이미지 전처리만 담당(리사이즈 224, CLIP 정규화). 텍스트는 아래에서 Qwen 토크나이저가 처리하므로 `CLIPProcessor` 전체가 필요 없습니다.
- `last_hidden_state[:, 1:]` — ViT의 출력은 `[B, 50, 768]`: 맨 앞이 CLS(요약) 토큰, 나머지 49개가 7×7 패치 각각의 벡터입니다. CLIP 레슨에서는 요약 벡터(`pooler_output`) 하나를 썼지만, 여기서는 **패치 49개를 전부** 씁니다. 벡터 하나로는 "어디에 무엇이 있는지"가 사라지기 때문입니다(CLIP 레슨 스스로 점검 Q3). 실습 2번에서 이 선택의 효과를 직접 비교합니다.
- `@torch.no_grad()` — 인코더는 얼릴 것이므로 기울기가 필요 없습니다.
- `ds[i:i + 64]["image"]` — 데이터셋을 슬라이스하면 열별 리스트를 주는 딕셔너리가 나옵니다. 64장씩 잘라 인코딩하는 이유는 6000장을 한 번에 GPU에 올릴 수 없기 때문입니다.
- `.half().cpu()` — 결과를 16비트 실수로 줄여 CPU 메모리에 보관. `6000 × 49 × 768 × 2바이트 ≈ 450MB`. 32비트면 900MB라서 절반으로 줄인 것이고, 학습 때 `.float()`으로 되돌립니다. 얼린 부품의 출력을 **미리 계산해 캐시**하는 것은 실무에서도 흔한 절약 기법입니다.
- `ds.remove_columns("image")` 뒤에 캡션만 리스트로 뽑는 이유 — `datasets`는 행에 접근할 때마다 이미지를 디코딩합니다. 캡션만 필요할 때 이미지 열을 빼 두면 수십 배 빠릅니다.
## 뇌: LLM (얼림)

```python
llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float32).to(device).eval()
for p in llm.parameters():
    p.requires_grad = False
embed = llm.get_input_embeddings()    # 토큰 번호 → 벡터. 미니 GPT 레슨의 tok_emb
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

**코드 읽기**

- `llm.get_input_embeddings()` — LLM의 토큰 임베딩 표(`nn.Embedding(151936, 896)`)를 꺼냅니다. 보통은 `input_ids`를 넣으면 모델 안에서 자동으로 이 표를 거치지만, 우리는 **이 표를 직접 불러** 텍스트를 벡터로 바꾼 뒤 이미지 벡터와 이어 붙여야 합니다. 그래서 따로 손에 쥡니다.
- `for p in llm.parameters(): p.requires_grad = False` — LLM 전체를 얼림. [CNN](#22-cnn) 레슨의 freeze와 같습니다. `.eval()`도 함께 둡니다(dropout 등이 있으면 꺼지도록).
- `nn.Sequential(Linear(768, 896), GELU(), Linear(896, 896))` — 2층 MLP 프로젝터. 왜 이 모양인가: LLaVA 1.5가 "Linear 하나보다 2층 MLP가 낫다"고 보고한 구조를 그대로 따랐습니다. 입력 768은 CLIP 패치 벡터의 차원, 출력 896은 Qwen 임베딩의 차원입니다. `vit.config.hidden_size`, `llm.config.hidden_size`에서 읽어 오므로 모델을 바꿔도 숫자를 고칠 필요가 없습니다.
- 학습할 파라미터 약 150만 개 — LLM(4.9억) + ViT(0.9억)의 0.3%도 안 됩니다. 이 작은 부품만으로 LLM이 이미지를 "읽게" 되는 것이 이 레슨의 요점입니다.
## 입력 조립: 텍스트 임베딩 사이에 이미지 끼워 넣기

채팅 형식의 프롬프트에서 `<image>` 자리를 기준으로 앞뒤를 나누고, 그 사이에 이미지 토큰을 넣습니다. 이번엔 토큰 번호(`input_ids`) 대신 **임베딩을 직접**(`inputs_embeds`) LLM에 넣는다는 점이 [LLM 파인튜닝](#33-llm-finetune) 레슨과 다릅니다.

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

**코드 읽기** — 이 함수가 VLM의 "조립 공정"입니다.

- `prompt.split("<image>")` — 채팅 템플릿으로 만든 프롬프트 문자열을 `<image>` 자리에서 앞(`pre`: 시스템 프롬프트 + `<|im_start|>user\n`)과 뒤(`post`: `\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n`)로 나눕니다. `<image>`는 특수 토큰이 아니라 그냥 우리가 정한 표시 문자열이고, 그 자리에 이미지 벡터가 들어갑니다.
- `pre_ids`, `post_ids`는 모든 샘플에 **똑같으므로** 함수 밖에서 한 번만 토큰화합니다.
- `seqs` — 캡션마다 토큰화하고 끝에 `eos_token`을 붙입니다([LLM 파인튜닝](#33-llm-finetune)과 같은 이유: 멈추는 법도 배워야 함). 길이가 제각각이라 `n = 최대 길이`를 구해 패딩합니다.
- `img = projector(img_feats.to(device).float())` — 캐시해 둔 16비트 특징을 32비트로 되돌려 프로젝터에 통과. `[B, 49, 768] → [B, 49, 896]`. 프로젝터는 마지막 축에만 작용하므로 49개 패치를 한 번에 처리합니다.
- `embed(ids([pre_ids] * B))` — 토큰 번호 → 임베딩 벡터. 배치 크기만큼 복제합니다.
- `torch.cat([pre_e, img, txt_e], dim=1)` — **시퀀스 축(dim=1)** 으로 이어 붙입니다. 결과 `[B, 앞 길이 + 49 + 뒤 길이 + 캡션 길이, 896]`. LLM 입장에서는 그냥 긴 임베딩 시퀀스이고, 그중 49개가 이미지에서 왔다는 것을 모릅니다. 이것이 "이미지를 처음 보는 단어처럼 읽는다"의 구현입니다.
- `labels` — 프롬프트와 이미지 위치는 `-100`(채점 제외), 캡션 부분만 실제 토큰, 패딩은 다시 `-100`. [LLM 파인튜닝](#33-llm-finetune)의 라벨 마스킹에 이미지 49칸이 추가된 것입니다.
- `mask` — 실제 내용이 있는 위치는 1, 패딩은 0. 이미지 토큰 49개도 1입니다(어텐션이 봐야 하므로).
- 마지막 `print`의 세 shape이 모두 같은 시퀀스 길이(97)를 가져야 합니다. 임베딩·마스크·라벨의 길이가 어긋나면 에러가 나거나, 더 나쁘게는 조용히 잘못 학습됩니다.
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

**코드 읽기**

- `describe(image)` — `build_batch`의 추론 버전. 캡션 없이 `pre + 이미지 + post`까지만 조립해서 LLM에게 "이어서 써 봐" 하는 것입니다. 배치 크기 1이라 패딩이 없어 마스크는 전부 1입니다.
- `llm.generate(inputs_embeds=embeds, attention_mask=mask, ...)` — `input_ids` 대신 `inputs_embeds`를 넘깁니다. Hugging Face 모델은 둘 중 하나를 받을 수 있고, 임베딩을 넘기면 임베딩 표를 건너뛰고 바로 트랜스포머 블록에 넣습니다. `inputs_embeds`로 생성하면 반환값에 프롬프트가 포함되지 않고 **새 토큰만** 돌아오므로 잘라 낼 필요가 없습니다.
- `show()` — 테스트 이미지 4장과 모델의 설명을 나란히 그립니다. 학습 전/후에 **같은 함수**를 불러 비교합니다. `ax.set_title(..., wrap=True)`는 긴 문장을 줄바꿈합니다.
- 학습 전 출력이 "이미지를 볼 수 없습니다"인 이유 — 프로젝터가 난수 상태라 이미지 토큰이 의미 없는 벡터이고, LLM은 텍스트 부분("이 이미지를 설명해줘")만 이해해 평소 하던 대답을 하는 것입니다.

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

**코드 읽기**

- `torch.optim.AdamW(projector.parameters(), lr=1e-3)` — 옵티마이저에 **프로젝터의 파라미터만** 넘깁니다. LLM과 ViT는 얼려 있으므로 애초에 후보가 아닙니다. 학습률 1e-3은 난수에서 새로 학습하는 층에 맞는 크기입니다(사전학습 가중치를 조정하는 것이 아니므로 LoRA보다 큼).
- `torch.randperm(len(feats))` — 0~5999를 무작위로 섞은 순서. 이것을 `batch_size`씩 잘라 배치를 만듭니다. `DataLoader` 대신 이렇게 하는 이유는 데이터가 이미 메모리의 텐서(`feats`)와 리스트(`captions`)로 준비되어 있어 인덱싱만 하면 되기 때문입니다.
- `random.choice(captions[j])` — 이미지마다 캡션이 5개 있으므로 에폭마다 다른 캡션을 뽑습니다. 같은 이미지에 표현이 다른 정답을 보여 주는 셈이라 **데이터 증강** 효과가 있습니다.
- `llm(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss` — LLM에 임베딩을 직접 넣고, 라벨을 주어 손실까지 받습니다. 얼린 LLM의 forward를 통과한 손실이지만, `embeds`는 프로젝터의 출력이므로 `loss.backward()`의 기울기가 LLM을 **거슬러 지나** 프로젝터까지 도달합니다. LLM의 파라미터는 `requires_grad=False`라 갱신되지 않을 뿐, 기울기가 통과하는 데는 문제가 없습니다.
- `step % 50 == 0` — 375스텝짜리 에폭에서 8번쯤 출력. loss가 5 → 2.5 근처로 내려가면 정상입니다.

## 학습 후

```python
show()
torch.save(projector.state_dict(), "projector.pt")   # 저장할 것은 프로젝터뿐 (약 6MB)
```

완벽하진 않습니다. 개·눈·잔디처럼 데이터에 자주 나온 장면은 잘 맞히지만, 사람 사진에는 "A man in a black shirt…" 같은 흔한 문장을 반복하기도 합니다(6000장은 아주 적은 데이터입니다). 그래도 학습 전의 "이미지를 볼 수 없습니다"와 비교해 보세요. **LLM도 이미지 인코더도 1바이트도 바꾸지 않았는데**, 작은 통역사 하나만 학습시켜서 LLM이 사진을 "읽게" 된 것입니다. 실제 VLM(LLaVA 등)은 이 1단계를 수십만 장으로 학습한 뒤, 2단계에서 LLM까지 함께 파인튜닝해 질문 답변 능력을 키웁니다.

## 핵심 정리

- VLM = **이미지 인코더(눈) + 프로젝터(통역사) + LLM(뇌)**.
- 이미지를 LLM의 임베딩과 같은 크기의 벡터들로 바꿔 텍스트 임베딩 사이에 끼워 넣습니다(`inputs_embeds`).
- 인코더와 LLM을 얼리고 프로젝터만 학습해도 LLM이 이미지를 '읽기' 시작합니다.
- 얼린 부분의 출력은 변하지 않으므로 미리 계산해 두면 학습이 빨라집니다.
- 기울기는 얼어 있는 LLM을 통과해 프로젝터까지 전달됩니다. 얼린다는 것은 '갱신하지 않는다'이지 '기울기가 못 지나간다'가 아닙니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 학습 전 모델이 '이미지를 볼 수 없습니다'라고 답하는 이유는?</summary>

무작위 프로젝터가 만든 벡터는 LLM에게 의미 없는 잡음입니다. LLM은 이미지 정보가 없는 상태에서 '이 이미지를 설명해줘'라는 텍스트만 본 셈이므로 그렇게 답합니다.

</details>

<details><summary>Q2. LLM의 가중치를 전혀 바꾸지 않았는데 어떻게 이미지를 이해하게 되었나요?</summary>

프로젝터가 이미지 특징을 'LLM이 이미 이해하는 임베딩 공간의 벡터'로 번역하도록 학습되었기 때문입니다. LLM 입장에서는 개·잔디 같은 단어와 비슷한 벡터가 들어온 것입니다.

</details>

<details><summary>Q3. 이미지 토큰 위치의 라벨을 -100으로 두는 이유는?</summary>

이미지 토큰은 예측해야 할 '다음 단어'가 없는 입력일 뿐입니다. 손실은 모델이 생성해야 하는 캡션 부분에서만 계산합니다.

</details>

## 직접 고쳐보기

1. 프로젝터를 `nn.Linear(768, 896)` 한 층으로 줄여보세요. loss와 설명 품질이 어떻게 달라지나요?
2. `image_features`에서 `[:, 1:]`을 `[:, :1]`로 바꿔 CLS 토큰 **1개만** 쓰도록 해보세요 (`feats`를 다시 계산해야 합니다). 이미지 토큰 49개 vs 1개의 차이는?
3. 직접 찍은 사진으로 시험해 보세요: Colab 왼쪽 파일 탭에 업로드 후 `from PIL import Image; describe(Image.open("내사진.jpg"))`. Flickr8k에 없는 종류의 사진(음식, 문서, 실내)에서는 어떻게 되나요? 왜일까요?
4. 프롬프트의 `"Describe this image."`를 바꿔서 학습/추론해 보세요.
5. (도전) 2단계 학습: [LLM 파인튜닝](#33-llm-finetune) 레슨의 LoRA를 `llm`에 붙이고, 프로젝터와 LoRA를 **함께** 학습시켜 보세요. 옵티마이저에 두 파라미터 그룹을 모두 넣어야 합니다.
6. (도전) 설명을 한국어로: `captions`를 번역(예: [LLM 파인튜닝](#33-llm-finetune) 레슨의 모델에게 시키기)해서 한국어 캡션 모델을 만들어 보세요.
