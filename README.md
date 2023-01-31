---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: zero-shot-image-classification
widget:
- src: https://huggingface.co/geolocal/StreetCLIP/resolve/main/nagasaki.jpg
  candidate_labels: China, South Korea, Japan, Phillipines, Taiwan, Vietnam, Cambodia 
  example_title: Countries
- src: https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg
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

Trained on an original dataset of 1.1 million street-level urban and rural geo-tagged images, it achieves
state-of-the-art performance on multiple open-domain image geolocalization benchmarks in zero-shot, 
outperforming supervised models trained on millions of images.

# Model Description

StreetCLIP is a model pretrained by deriving image captions synthetically from image class labels using
a domain-specific caption template. This allows StreetCLIP to transfer its generalized zero-shot learning
capabilities to a specific domain (i.e. the domain of image geolocalization). 
StreetCLIP builds on the OpenAI's pretrained large version of CLIP ViT, using 14x14 pixel
patches and images with a 336 pixel side length.

## Model Details

- **Model type:** [CLIP](https://openai.com/blog/clip/)
- **Language:** English
- **License:** Create Commons Attribution Non Commercial 4.0
- **Trained from model:** [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

## Model Sources

- **Paper:** Pre-print available soon ...

# Uses

StreetCLIP has a deep understanding of the visual features found in street-level urban and rural scenes
and knows how to relate these concepts to specific countries, regions, and cities. Given its training setup,
the following use cases are recommended for StreetCLIP.

## Direct Use

StreetCLIP can be used out-of-the box using zero-shot learning to infer the geolocation of images on a country, region,
or city level. Given that StreetCLIP was pretrained on a dataset of street-level urban and rural images,
the best performance can be expected on images from a similar distribution.

Broader direct use cases are any zero-shot image classification tasks that rely on urban and rural street-level
understanding or geographical information relating visual clues to their region of origin.

## Downstream Use

StreetCLIP can be finetuned for any downstream applications that require geographic or street-level urban or rural
scene understanding. Examples of use cases are the following:

**Understanding the Built Environment**

- Analyzing building quality
- Building type classifcation
- Building energy efficiency Classification

**Analyzing Infrastructure**

- Analyzing road quality
- Utility pole maintenance
- Identifying damage from natural disasters or armed conflicts

**Understanding the Natural Environment**

- Mapping vegetation
- Vegetation classification
- Soil type classifcation
- Tracking deforestation

**General Use Cases**

- Street-level image segmentation
- Urban and rural scene classification
- Object detection in urban or rural environments
- Improving navigation and self-driving car technology

## Out-of-Scope Use

Any use cases attempting to geolocate users' private images are out-of-scope and discouraged.

# Bias, Risks, and Limitations

StreetCLIP was not trained on social media images or images of identifable people for a reason. As such, any use case
attempting to geolocalize users' private images 

## Recommendations
We encourage the community to apply StreetCLIP to applications with significant social impact of which there are many.
The first three categories of potential use cases under Downstream Use list potential use cases with social impact
to explore.

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

StreetCLIP was trained on an original, unreleased street-level dataset of 1.1 million real-world,
urban and rural images. The data used to train the model comes from 101 countries, biased towards
western countries and not including India and China.

## Preprocessing

Same preprocessing as [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336).

## Training Procedure

StreetCLIP is initialized with OpenAI's pretrained large version of CLIP ViT and then pretrained using the synthetic
caption domain-specific pretraining method described in the paper corresponding to this work. StreetCLIP was trained
for 3 epochs using an AdamW optimizer with a learning rate of 1e-6 on 3 NVIDIA A100 80GB GPUs, a batch size of 32,
and gradient accumulation of 12 steps.

StreetCLIP was trained with the goal of matching images in the batch
with the caption correponding to the correct city, region, and country of the images' origins.

# Evaluation

StreetCLIP was evaluated in zero-shot on two open-domain image geolocalization benchmarks using a
technique called hierarchical linear probing. Hierarchical linear probing sequentially attempts to
identify the correct country and then city of geographical image origin.

## Testing Data and Metrics

### Testing Data

StreetCLIP was evaluated on the following two open-domain image geolocalization benchmarks.

* [IM2GPS](http://graphics.cs.cmu.edu/projects/im2gps/).
* [IM2GPS3K](https://github.com/lugiavn/revisiting-im2gps)

### Metrics

The objective of the listed benchmark datasets is to predict the images' coordinates of origin with as
little deviation as possible. A common metric set forth in prior literature is called Percentage at Kilometer (% @ KM).
The Percentage at Kilometer metric first calculates the distance in kilometers between the predicted coordinates
to the ground truth coordinates and then looks at what percentage of error distances are below a certain kilometer threshold.

## Results

**IM2GPS**

| | Distance (% @ km) |
| Model | City | Region | Country | Continent |
|   |  25km | 200km  | 750km | 2,500km |
|----------|:-------------:|:------:|:------:|:------:|
| PlaNet (2016) |  24.5 | 37.6 | 53.6 | 71.3 |
| ISNs (2018) |  43.0 | 51.9 | 66.7 | 80.2 |
| TransLocator (2022) |  **48.1** | **64.6** | **75.6** | 86.7 |
| **Zero-Shot CLIP (ours)** | 27.0 | 42.2 | 71.7 | 86.9 |
| **Zero-Shot StreetCLIP (ours)** |  28.3 | 45.1 | 74.7 | **88.2** |


### Summary

Our experiments demonstrate that our synthetic caption pretraining method is capable of significantly
improving CLIP's generalized zero-shot capabilities applied to open-domain image geolocalization while
achieving state-of-the-art performance on a selection of benchmark metrics.

# Environmental Impact

- **Hardware Type:** 4 NVIDIA A100 GPUs
- **Hours used:** 12

# Citation

Preprint available soon ...

**BibTeX:**

Available soon ...
