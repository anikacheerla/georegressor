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
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# Remove text portion of model, only pretrain on image
# for param in model.text_model.parameters():
#     param.requires_grad = False

# TODO(Anika): Could try vice versa 
# num_classes = 100
# model.fc = nn.Linear(model.vision_model.config.hidden_size, num_classes)

# Load dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = np.loadtxt("training_data.csv", str, delimiter=',')
choices, counts = np.unique(dataset[:, 1], return_counts=True)
# choices = list(choices)

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, prompt = self.data.iloc[idx]
        image = Image.open(image_path)
        return F.to_tensor(image).to(device), prompt

train_dataset = CustomDataset("training_data.csv")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.to(device)
model.train()

epochs = 5

i = 0

prompt_to_label_id = {prompt : i for prompt, i in zip(choices, range(len(choices)))}
print(prompt_to_label_id)
for e in range(epochs):
    epoch_loss = 0.0

    for images, prompts in tqdm(train_loader):
        # images = [img for img in images]
        inputs = processor(text=list(choices), images=images, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
    
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        
        # Find the index of the prompt label in choices
        # Create the target tensor with the same shape as logits_per_image
    
        # Now, compute the loss using the target tensor
        # print(choices)
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print([torch.argmax(prob) for prob in probs])
        print([prompt_to_label_id[prompt] for prompt in prompts])
        loss = loss_fn(logits_per_image, torch.tensor([prompt_to_label_id[prompt] for prompt in prompts]).to(device))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(images)

        if i % 4 == 0:
            print(f"Loss: {loss.item()}")
    model.save_pretrained("fine_tuned_streetclip_frozen")
    print(f"Epoch Loss: {epoch_loss / len(train_dataset)}")

    # Save the fine-tuned model
model.save_pretrained("fine_tuned_streetclip_frozen")