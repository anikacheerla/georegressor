import torch
import numpy as np
from PIL import Image
import pandas as pd

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm

# Options:
# Default is just CLIP, train with new captions and don't freeze text encoding
# Default is just CLIP, traing with only country names and freeze text encoding
# Default is StreetClip, train with only country names and freeze text encoding
# Default is CLIP, train with only country names and freeze text encoding

# Load pre-trained model
# Load pre-trained model
# model = CLIPForImageClassification.from_pretrained("geolocal/StreetCLIP")
# model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
# processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # openai/clip-vit-base-patch32
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# freeze the model
# for param in model.text_model.parameters():
#     param.requires_grad = False
    

# freeze text parameters
# Load dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = pd.read_csv("training_data.csv", header=None)
classes = dataset[1].unique()
shorter_classes = np.array([c.partition('.')[0] + '.' for c in classes])
labels =  np.array([c.partition('.')[0].split()[-1] for c in classes])



class CustomDataset(Dataset):
    def __init__(self, data_path, processor, classes):
        self.data = pd.read_csv(data_path, header=None)
        self.processor = processor
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, prompt = self.data.iloc[idx]
        # prompt = prompt.partition('.')[0].split()[-1]
        label = np.where(self.classes == prompt)[0][0]  # Find label index
        image = Image.open(image_path)
        return F.pil_to_tensor(image).to(device), torch.tensor(label).to(device)


train_dataset = CustomDataset("training_data.csv", processor, classes) # special captions
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.to(device)
model.train()

epochs = 2

i = 0

prompt_to_label_id = {prompt : i for prompt, i in zip(classes, range(len(classes)))}
# print(prompt_to_label_id)
for e in range(epochs):
    epoch_loss = torch.tensor(0.0)

    for images, prompts in tqdm(train_loader):
        # images = [img for img in images]
        inputs = processor(text=list(classes), images=images, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        
        # Find the index of the prompt label in choices
        # Create the target tensor with the same shape as logits_per_image
    
        # Now, compute the loss using the target tensor
        # print(choices)
        probs = logits_per_image.softmax(dim=1).to(device) # we can take the softmax to get the label probabilities
        loss = loss_fn(logits_per_image, prompts).to(device)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        i+=1
        if i % 100 == 0:
            print(f"Loss: {loss}")
    model.save_pretrained("fine_tuned_clip_captions")
    print(f"Epoch Loss: {epoch_loss / len(train_dataset)}")

    # Save the fine-tuned model
model.save_pretrained("fine_tuned_clip_captions")