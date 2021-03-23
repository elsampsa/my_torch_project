# as per: https://pytorch.org/hub/pytorch_vision_resnet/

# sample execution (requires torchvision)
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

#model = models.resnet50()
#pth_file = "../tmp/resnet50-19c8e357.pth" # https://download.pytorch.org/models/resnet50-19c8e357.pth

model = models.resnet18()
pth_file = "../tmp/resnet18-5c106cde.pth"

image_file="data/dog.jpg" # https://github.com/pytorch/hub/raw/master/images/dog.jpg

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model.load_state_dict(
    torch.load(pth_file, map_location = device)
)

input_image = Image.open(image_file)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
print("max>", probabilities.max())
# Read the categories
with open("data/imagenet_classes.txt", "r") as f: # https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
