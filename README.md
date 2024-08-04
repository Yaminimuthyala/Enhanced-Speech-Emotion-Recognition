# Enhanced Speech Emotion Recognition: Leveraging Non-Linear Deep Learning Models (CNN, DNN, LSTM, Auto Encoders) with Data Augmentation
## Description
In human-computer interaction, recognizing emotions in speech through deep learning is challenging yet essential for enhancing user experience. Traditional methods, using MFCC and chroma features with sigmoid and softmax activation functions, typically achieve 70-75% accuracy. This study improves results by incorporating a broader range of features like root mean square, mel spectrograms, Zero-Crossing Rate (ZCR), and data augmentation, alongside standard MFCC and chroma features. Adopting the Rectified Linear Unit (ReLU) activation function and employing various deep learning architectures, including LSTM, CNN, DNN, and Autoencoders, we achieved notable results. The Autoencoders + 1-D CNN model, optimized with hyperparameters and ReLU, reached the highest accuracy at 86%, surpassing LSTM (83%), CNN (83%), and DNN (79%). These findings demonstrate the effectiveness of diverse features, alternative activation functions, and rigorous hyperparameter tuning in speech emotion recognition, suggesting new paths for enhancing human-computer interaction technologies.
## Dataset
RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song.

SAVEE: Surrey Audio-Visual Expressed Emotion.

CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.

## Methodology
### Data Collection
Google Drive Integration: Mounted Google Drive to access datasets stored in the drive.

Path Definitions: Defined paths for RAVDESS, SAVEE, and CREMA-D datasets.
### Data Preprocessing
**Data Loading**: Loaded audio files from the specified paths.

**Feature Extraction**: Extracted features such as MFCCs, root mean square, mel spectrograms, and Zero-Crossing Rate (ZCR).

**Data Augmentation**: Applied techniques like noise addition to augment the dataset.

**Data Merging and Cleaning**: Merged datasets from multiple sources and ensured data quality.

**Data Splitting**: Split the data into training, validation, and testing sets to ensure robust model evaluation.

### Modeling
**DNN (Dense Neural Network)**: Implemented DNN models for baseline emotion classification, using dense layers and dropout for regularization.

**RNN (Recurrent Neural Network)**: Utilized RNNs to capture temporal dependencies in the audio data, specifically LSTM layers.

**CNN (Convolutional Neural Network)**: Employed CNNs for feature extraction from spectrogram images, using convolutional and pooling layers.

**Autoencoders and Decoders**: Developed autoencoders for unsupervised learning and feature extraction, using decoders to reconstruct input data from encoded representations.

### Training and Evaluation

**Optimization**: Trained models using the Adam optimizer with learning rate scheduling.

**Activation Function**: Adopted ReLU activation function for better performance.

**Hyperparameter Tuning**: Conducted extensive hyperparameter tuning to optimize model performance.

**Performance Metrics**: Evaluated models using accuracy, precision, recall, and F1-score metrics.

### File Descriptions

**Data_preprocess.ipynb**: Contains the initial data preprocessing steps, feature extraction, and preliminary data augmentation techniques.

**Modelling.ipynb**: Includes the implementation and training of various models (DNN, RNN, CNN, Autoencoders), hyperparameter tuning, and model evaluation.
