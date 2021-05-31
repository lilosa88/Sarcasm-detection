# Sarcasm-detection
# Objective

- This project belongs to [kaggle's competitions](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- We will make use of a public data-sets published by [Rishabh Misra](https://rishabhmisra.github.io/publications/) with details on [Kaggle] (https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home). News Headlines dataset for Sarcasm Detection is collected from two news website. [TheOnion](https://www.theonion.com/) aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from [HuffPost](https://www.huffingtonpost.com/). 


# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, re, nltk, json, keras, tensorflow

# Data description

Each record consists of three attributes:

- is_sarcastic: 1 if the record is sarcastic otherwise 0

- headline: the headline of the news article

- article_link: link to the original news article. Useful in collecting supplementary data

This dataset has following advantages

- Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.

- Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels.

- The headlines we obtained are self-contained. This would help us in teasing apart the real sarcastic elements.

# Preprocessing

- We carried out the pre-processing with the following hyperparameters:
  - vocab_size = 1000 
  - embedding_dim = 32 
  - max_length = 16 # and this(it was 32)
  - trunc_type='post'
  - padding_type='post'
  - oov_tok = "<OOV>"
  - training_size = 20000

- We clean the data by removing punctuations, stopwords and applying lowercase. Thus we use PorterStemmer, stemming is the process of reducing words to their word stem.
- We convert our sentences into vectors using Bag of words model.
- We applying encoding into the column label.
- Train and test split. 

# Machine Learning Models

- Naive Bayes Model
 
 Train Random Forest's Accuracy:  0.9887
 
 Test Random Forest's Accuracy:  0.9838
 
