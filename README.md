# Dog vs. Cat Classification Using CNN 

This project demonstrates how to build a Convolutional Neural Network (CNN) for binary classification (dog vs. cat) using images and their corresponding XML annotations. The dataset is organized into two directories: one for images and one for annotations. The annotations (in XML format) are used to extract image filenames and their labels.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- TensorFlow (and Keras)
- scikit-learn

Install the required libraries using pip:

```bash
pip install opencv-python numpy pandas matplotlib tensorflow scikit-learn
```

## Code Overview

The main steps in the code are as follows:

### Dataset Preparation

- **Paths Setup:** Define paths to the images and annotations folders.
- **XML Parsing:** For each XML file, the code reads the image filename and label from the XML (the `<filename>` and `<object>/<name>` tags).
- **Image Preprocessing:** Images are read in grayscale using OpenCV, resized to 80×80 pixels, and normalized.
- **Label Conversion:** The function `convert_label` converts the string label (e.g., `"dog"`) into a numerical value (1 for dog, 0 for cat).

### Data Splitting

- The dataset is split into training and testing sets using `train_test_split` from scikit-learn (80% training, 20% testing).

### CNN Model Construction

- A Sequential CNN model is built with two convolutional layers, each followed by max-pooling, then a flattening layer, and two dense layers.
- The final output layer uses a sigmoid activation function for binary classification.
- The model is compiled with the Adam optimizer and binary cross-entropy loss.

### Model Training and Evaluation

- The model is trained for 13 epochs with a batch size of 32.
- After training, the model is evaluated on the test set to compute the accuracy.
- Predictions are generated on the test set, and the predicted values are rounded to obtain binary classification results.
