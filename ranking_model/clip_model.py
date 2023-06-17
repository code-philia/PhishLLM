import os

import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
ct = 0
for path in sorted(os.listdir("./datasets/alexa_login/taobao.com")):
    ct += 1
    image = preprocess(Image.open("./datasets/alexa_login/taobao.com/{}".format(path))).unsqueeze(0).to(device)
    button_cls = ["login button", "signup button", "other button"]
    text = clip.tokenize(button_cls).to(device)
    zeroshot_text_inputs = torch.cat([clip.tokenize(f"a {c}") for c in button_cls]).to(device)
    # align
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
    #
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #     pred = logits_per_image.argmax(dim=-1).cpu().numpy()[0]
    #     conf = probs[0, pred]
    #
    #     plt.subplot(3, 3, ct)
    #     plt.imshow(Image.open("./datasets/alexa_login/taobao.com/{}".format(path)))
    #     plt.title('Pred={}, Conf={:.3f}'.format(button_cls[pred], conf))
    #     if ct == 9:
    #         plt.show()
    #         exit()
    #
    # print(path)
    # print("Label probs:", probs)
    # print("Label pred:", pred)


    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(zeroshot_text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    # Print the result
    print(path)
    print("Top predictions:")
    for value, index in zip(values, indices):
        print(f"{button_cls[index]:>16s}: {100 * value.item():.2f}%")
