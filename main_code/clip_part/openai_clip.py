import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)


image = preprocess(Image.open("../../temp_images/image030.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a person is climbing over a metal gate","a person is passing through a metal gate"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = list(logits_per_image.softmax(dim=-1).cpu().numpy())
    clean_result = []
    for img_result in probs:
        clean_result.append([round(i,2) for i in img_result])


print("Label probs:", clean_result)  # prints: [[0.9927937  0.00421068 0.00299572]]