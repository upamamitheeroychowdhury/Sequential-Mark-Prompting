# Visual question answering with grounding and user selection priority

### Introduction

Enables users to query images using text input. Users can select specific objects for their queries, streamlining the process of asking questions and eliminating the need to describe the position of objects within the image using spatial words.

### Architecture
<p align="center">
  <img src="https://github.com/shetumohanto/mistral/assets/53278488/ba3c39fe-042b-44a9-8726-d1d7c154f029">
</p>

### 🔗 System requirements
* Requires a `CUDA compatible GPU` with minimum `8gb VRAM`
* `Python>=3.8`

### :rocket: Quick Start
* Add your `GPT-4 Vision API Key` using a `.env` file defined as `OPENAI_API_KEY`
```bash
from dotenv import load_dotenv
load_dotenv()
```

* Clone the repository
```bash
git clone https://github.com/shetumohanto/visual-question-answering.git
cd visual-question-answering
```

* Download checkpoint for the `segment anything` `vit_h` model. 
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

* Install required dependencies
```bash
pip install -r requirements.txt
```
* Run Intelligent EYE
```bash
streamlit run app.py
```
### 🔗 Related Works

This project uses the following technologies at its core architecture:
- [Segment Anything](https://github.com/facebookresearch/segment-anything): Segment anything
- [SoM](https://github.com/microsoft/SoM): Set-of-Mark Visual Prompting for GPT-4V

### 🔗 Sample output
![example](https://github.com/shetumohanto/mistral/assets/53278488/c652ec68-4514-4be3-8b6c-17db04d37fc7)
