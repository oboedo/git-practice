from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 모델 로드
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 이미지와 텍스트 입력 정의
image = Image.open("example.jpg")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 전처리 및 모델 추론
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 이미지와 텍스트 유사도
probs = logits_per_image.softmax(dim=1)  # 확률 계산

# 결과 출력
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
