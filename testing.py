from PIL import Image
import requests
import numpy as np

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

def process_batch(images, prompts):
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    predictions = [choices[np.argmax(prob.detach().numpy())] for prob in probs]
    return predictions

dataset = np.loadtxt("small_testing_dataset_with_prompts.csv", str, delimiter=',')
choices, counts = np.unique(dataset[:, 1], return_counts=True)
prompt_successes = {prompt: 0 for prompt in choices}

batch_size = 16
num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)

for batch_idx in range(170, num_batches): # already did first 38 of singapore, 3 were wrong
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(dataset))
    
    batch_images = [Image.open(image_path) for image_path, _ in dataset[start_idx:end_idx]]

    predictions = process_batch(batch_images, list(choices))
    
    for pred, (_, prompt) in zip(predictions, dataset[start_idx:end_idx]):
        if pred == prompt:
            prompt_successes[prompt] += 1
        print("Prediction", pred, "Prompt", prompt)

for prompt, success_count in prompt_successes.items():
    country = list(prompt.split(" "))
    success_percentage = success_count / counts[np.where(choices == prompt)[0]]
    print(country, success_percentage)