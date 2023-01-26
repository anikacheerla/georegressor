---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: zero-shot-image-classification
widget:
- src: https://huggingface.co/lhaas/StreetCLIP/resolve/main/nagasaki.jpg
  candidate_labels: China, South Korea, Japan, Phillipines, Taiwan, Vietnam, Cambodia 
  example_title: Countries
- src: https://huggingface.co/lhaas/StreetCLIP/resolve/main/sanfrancisco.jpeg
  candidate_labels: San Jose, San Diego, Los Angeles, Las Vegas, San Francisco, Seattle
  example_title: Cities
library_name: transformers
tags:
- geolocalization
- geolocation
- geographic
- street
- climate
- clip
- urban
- rural
- multi-modal
---
# Model Card for StreetCLIP

StreetCLIP is a robust foundation model for open-domain image geolocalization and other
geographic and climate-related tasks.

Trained on a dataset of 1.1 million geo-tagged images, it achieves state-of-the-art performance
on multiple open-domain image geolocalization benchmarks in zero-shot, outperforming supervised models
trained on millions of images.


# Model Details

## Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:** Authors not disclosed
- **Model type:** [CLIP](https://openai.com/blog/clip/)
- **Language:** English
- **License:** Create Commons Attribution Non Commercial 4.0
- **Finetuned from model:** [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

## Model Sources

- **Paper:** Pre-print available soon ...
- **Demo:** Currently in development ...

# Uses

To be added soon ...

## Direct Use

To be added soon ...

## Downstream Use

To be added soon ...

## Out-of-Scope Use

To be added soon ...

# Bias, Risks, and Limitations

To be added soon ...

## Recommendations

To be added soon ...

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

choices = ["San Jose", "San Diego", "Los Angeles", "Las Vegas", "San Francisco"]
inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```

# Training Details

## Training Data

StreetCLIP was trained on an undisclosed street-level dataset of 1.1 million real-world,
urban and rural images. The data used to train the model comes from 101 countries.

## Training Procedure

### Preprocessing

Same preprocessing as [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336).

# Evaluation

StreetCLIP was evaluated in zero-shot on two open-domain image geolocalization benchmarks using a
technique called hierarchical linear probing. Hierarchical linear probing sequentially attempts to
identify the correct country and then city of geographical image origin.

## Testing Data, Factors & Metrics

### Testing Data

* [IM2GPS](http://graphics.cs.cmu.edu/projects/im2gps/).
* [IM2GPS3K](https://github.com/lugiavn/revisiting-im2gps)

### Metrics

To be added soon ...

## Results

To be added soon ...

### Summary

Our experiments demonstrate that our synthetic caption pretraining method is capable of significantly
improving CLIP's generalized zero-shot capabilities applied to open-domain image geolocalization while
achieving SOTA performance on a selection of benchmark metrics.

# Environmental Impact

- **Hardware Type:** 4 NVIDIA A100 GPUs
- **Hours used:** 12

# Example Image Attribution

To be added soon ...

# Citation

Preprint available soon ...

**BibTeX:**

Available soon ...
