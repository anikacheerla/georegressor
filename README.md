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
- src: https://huggingface.co/lhaas/StreetCLIP/resolve/main/australia.jpeg
  candidate_labels: tropical climate, dry climate, temperate climate, continental climate, polar climate
  example_title: Climate
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
---
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->


# Model Details

## Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:** Authors not disclosed
- **Model type:** [CLIP](https://openai.com/blog/clip/)
- **Language:** English
- **License:** Create Commons Attribution Non Commercial 4.0
- **Finetuned from model:** [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

## Model Sources

<!-- Provide the basic links for the model. -->

- **Paper:** Pre-print available soon ..
- **Demo:** Currently in development ...

# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

## Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("lhaas/StreetCLIP")
processor = CLIPProcessor.from_pretrained("lhaas/StreetCLIP")

url = "https://huggingface.co/lhaas/StreetCLIP/resolve/main/sanfrancisco.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

choices = ["San Jose", "San Diego", "Los Angeles", "Las Vegas", "San Francisco"]
inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```

# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

## Training Procedure [optional]

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

### Preprocessing

[More Information Needed]

### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

[More Information Needed]

### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

## Results

[More Information Needed]

### Summary



# Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

# Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** 4 NVIDIA A100 GPUs
- **Hours used:** 12

# Example Image Attribution

[More information needed]

# Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]
