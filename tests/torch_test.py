import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading ImageNet Labels
import urllib.request, json
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(urllib.request.urlopen(url).read())

# Inference
# it's a rough basic idea to implement imputer in torch. probably not the ideal way to do prediction here,
# if we want to test the imputer functionality.
# of course it shouldn't all are packed within predict function.
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    # imputer should be somewhere here to process img_t.

    # probbably dataloader is here.
    
    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    
    # Top 5
    top5 = torch.topk(probs, 5)
    print(f"\n The classification result of {image_path}:")
    for i in range(5):
        print(f"{labels[top5.indices[i]]}: {top5.values[i]:.2%}")

# test
predict("tests/pics/cat.jpg")
predict("tests/pics/dog.png")