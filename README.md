# Enhancing-Image-Captioning-with-CNN-and-Transformer-Model

## Overview
This project implements an image captioning model using a Transformer-based encoder-decoder architecture and EfficientNet as the CNN backbone for extracting image features. The dataset used is Flickr8k, and the model predicts captions for input images. The project includes data preprocessing, model training, and caption generation with evaluation metrics such as BLEU score.

---

## Features
- **Data Preprocessing:** Downloads and preprocesses the Flickr8k dataset, including image resizing and caption tokenization.
- **Image Augmentation:** Enhances images using random flips, rotations, and contrast adjustments.
- **EfficientNet Backbone:** Uses EfficientNetB1 for feature extraction from images.
- **Transformer Architecture:** Implements positional embeddings, attention mechanisms, and dense layers for caption generation.
- **BLEU Metric Integration:** Custom BLEU metric for evaluation of generated captions.
- **Learning Rate Scheduling:** Implements a warm-up learning rate scheduler for efficient training.

---

## Dataset
The project uses the Flickr8k dataset, which consists of:
- **Images:** JPEG files in the `Flicker8k_Dataset` directory.
- **Captions:** Descriptive sentences in `Flickr8k.token.txt`.

### Dataset Preparation
Run the `download_and_unzip` function to download and extract the dataset. Captions are filtered to exclude those with fewer than 5 or more than 25 words.

---

## Dependencies
Install the required packages:

```bash
pip install tensorflow keras numpy matplotlib nltk
```

---

## Project Structure
- **Data Preparation:**
  - `load_captions_data`: Loads and filters captions.
  - `train_val_split`: Splits the data into training and validation sets.

- **Model Components:**
  - `get_cnn_model`: EfficientNet-based feature extractor.
  - `TransformerEncoderBlock`: Transformer encoder layer for image features.
  - `TransformerDecoderBlock`: Transformer decoder for generating captions.
  - `ImageCaptioningModel`: Combines the CNN and Transformer components for training and prediction.

- **Evaluation:**
  - `calculate_bleu_score`: Evaluates generated captions using BLEU scores.
  - `BLEUMetric`: Custom Keras metric for tracking BLEU scores during training.

---

## Training
### Steps to Train the Model
1. Load and preprocess the dataset.
2. Create training and validation datasets using `make_dataset`.
3. Initialize the model components:
   - CNN Model (`EfficientNetB1`)
   - Encoder and Decoder (Transformer blocks)
4. Compile the model with:
   - **Loss:** Sparse Categorical Crossentropy
   - **Optimizer:** Adam with warm-up learning rate scheduler
   - **Metrics:** Accuracy and BLEU Score
5. Train the model using the `fit` method with early stopping.

### Example:
```python
caption_model.fit(
    train_dataset,
    epochs=50,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
```

---

## Caption Generation
The `generate_caption` function:
1. Selects a random image from the validation set.
2. Extracts image features using the CNN model.
3. Generates captions token by token using the decoder.
4. Displays the image and predicted caption.

### Example:
```python
generate_caption()
```

---

## Customization
- Modify **Transformer Encoder/Decoder** parameters:
  - Number of heads in multi-head attention.
  - Embedding dimensions.
  - Feedforward dimensions.
- Change the **CNN backbone** to another EfficientNet variant (e.g., `EfficientNetB0` or `EfficientNetB7`).

---

## Results
The model outputs captions like:
- Input Image: Displays a random image from the validation set.
- Predicted Caption: A human-readable caption describing the image.

Example Output:
```plaintext
Predicted Caption: a dog playing with a ball on grass
```

---

## Limitations
- Limited to the Flickr8k dataset; larger datasets may improve performance.
- BLEU scores depend on the quality and variety of captions in the dataset.

---

## Future Work
- Experiment with larger datasets (e.g., COCO).
- Implement beam search for improved caption quality.
- Fine-tune EfficientNet layers for better feature extraction.
- Explore advanced evaluation metrics like CIDEr or METEOR.

---

## References
- **Flickr8k Dataset:** [https://github.com/jbrownlee/Datasets](https://github.com/jbrownlee/Datasets)
- **EfficientNet:** Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
- **BLEU Metric:** Papineni, K., et al. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation.

