# Speech Emotion Recognition
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model](#model)
- [Results](#results)
- [Paper Implementation](#paper-implementation)
- [Notebooks](#notebooks)
- [Collaborators](#collaborators)

## Introduction
Speech is the most natural way of expressing ourselves as humans. It is only natural
then to extend this communication medium to computer applications. We define
speech emotion recognition (SER) systems as a collection of methodologies that
process and classify speech signals to detect the embedded emotions. 

This project was a practical introduction to CNNs for feature extraction and classification of speech signals. We applied both 1D and 2D CNNs to get a better understanding of the differences between the two. 

We also closely followed the paper [Convolutional Neural Networks (CNN) Based Speech-Emotion Recognition](10.1109/SPICSCON48833.2019.9065172) by [Alif Bin Abdul Qayyum](https://www.researchgate.net/profile/Alif-Bin-Abdul-Qayyum), [Asiful Arefeen](https://www.researchgate.net/profile/Asiful-Arefeen), and [Celia Shahnaz](https://scholar.google.com/citations?user=gwrHyr8AAAAJ&hl=en).

## Dataset
We used the [CREMA dataset](https://www.kaggle.com/datasets/ejlok1/cremad) which is a collection of 7,442 original clips. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified). 

In our paper implementation, we also utilised the [SAVEE dataset](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en). The dataset was recorded from four native English male speakers (identified as DC, JE, JK, KL), postgraduate students and researchers at the University of Surrey aged from 27 to 31 years. Emotion has been described psychologically in discrete categories: anger, disgust, fear, happiness, sadness and surprise. 

## Preprocessing
The first step in our pipeline was to preprocess the audio files. We used the [librosa](https://librosa.org/doc/latest/index.html) library to load the audio files and extract the audio features. We then used the [scikit-learn](https://scikit-learn.org/stable/) library to split the dataset into training and testing sets. We also used augmentation to increase the size of the training set though it did not improve the accuracy of the model.

## Feature Extraction
We used the [librosa](https://librosa.org/doc/latest/index.html) library to extract the audio features as well. We extracted the following features for the 1D CNN:
- Mel-frequency cepstral coefficients (MFCC) (20, 100 coefficients)
- Mel-spectrogram 
- Zero-crossing rate (ZCR)
- Energy 
- Chroma frequencies 
- Spectral bandwidth 
- Spectral rolloff 
- Spectral contrast 
- Spectral flatness 
- Tonnetz 
as well as many other features that were eliminated during the incremental feature selection process.

We only extracted the Mel-spectrogram for the 2D CNN.

For the paper implementation, we extracted modulation spectral features and MFCC.

## Model
We used the [Keras](https://keras.io/) library to build our CNN models. We experimented with many models with the best testing accuracy being 0.48 for the 1D CNN (Model 2 using LDA) and 0.55 for the 2D CNN (CNN MODEL2).


## Results
The following table shows the results of our experiments with the 1D CNN. The best testing accuracy was 0.48 for Model 2 using LDA.

| Model | Validation Accuracy | Testing Accuracy | Testing F1-Score |
| --- | --- | --- | --- |
| Model 1 | 0.54 | 0.48 | 0.47 |
| Model 2 | 0.55 | 0.48 | 0.48 |
| Model 3 | 0.55 | 0.46 | 0.46 |
| Model 4 | 0.53 | 0.46 | 0.46 |
| Model 5 | 0.40 | 0.38 | 0.36 |
| Model 6 | 0.46 | 0.45 | 0.43 |
| Model 3 with augmentation | 0.49 | 0.47 | 0.47 |

The following table shows the results of our experiments with the 2D CNN. The best testing accuracy was 0.55 for CNN MODEL2.

| Model | Validation Accuracy | Testing Accuracy | Testing F1-Score |
| --- | --- | --- | --- |
| Initial Model | 0.44 | 0.41 | 0.40 |
| CNN MODEL2 | 0.58 | 0.55 | 0.56 |
| CNN MODEL3 | 0.42 | 0.40 | 0.33 |

As for the paper implementation, on the SAVEE dataset we got an accuracy of 60%, but not a significant improvement on the CREMA dataset with an accuracy of 48%.
It’s also worth noting that the SAVEE dataset was recorded from 4 native english male speakers
aged 27 to 31 years. On the other hand, CREMA-D is from 91 actors, 48 male, 43 female, ages
20 to 74, from a variety of races and ethnicities.
This would make it easier to achieve higher accuracies using the SAVEE dataset.

## Paper Implementation
A crucial detail in the paper was the MSF extraction process. We suspect the large disparity in the results between our implementation and the paper's is due to a different implementation of the MSF extraction process. 
We used SAVEE dataset to be able to compare our results to the paper's. We used the same model architecture as the paper and got an accuracy of 60% on the SAVEE dataset. We also tried to use the CREMA dataset but we got an accuracy of 48% which is not a significant improvement over our previous results.

## Notebooks
- [1D CNN](./SER_assignment3.ipynb)
- [2D CNN](./SER_assignment3_2D.ipynb)
- [Paper Implementation](./SER_assignment3_paper.ipynb)

## Collaborators
- [Manar Amgad](https://github.com/manaramgadd)
- Ahmed Dusuki