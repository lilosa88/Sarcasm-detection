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

- Test and train split: This dataset has about 27,000 records. So, we train on 20,000 and validate on the rest. 

-For both train and test data we apply the following steps:

  - We carried out the pre-processing with the following hyperparameters:
    - vocab_size = 1000 
    - embedding_dim = 32 
    - max_length = 16 # and this(it was 32)
    - trunc_type='post'
    - padding_type='post'
    - oov_tok = "<OOV>"
    - training_size = 20000
  
  - First, we apply tokenizer which is an encoder ofer by Tensorflow and keras. This works by generating a dictionary of word encodings and creating vectors out    of the sentences. The hyper-parameter vocab_size is given as the number of words. So by setting this hyperparameter, what the tokenizer will do is take the      top number of words given in vocab_size and just encode those. On the other hand, in many cases, it's a good idea to instead of just ignoring unseen words, to    put a special value when an unseen word is encountered. You can do this with a property on the tokenizer. This property is oov token and is set in the            tokenizer constructor. I've specified that I want the token OOV for outer vocabulary to be used for words that aren't in the word index.
  
  - Second, we apply the fit_on_texts method of the tokenizer that actually encodes the data following the hyper-parameter given previosuly. 
  
  - Third, we apply the word_index method. The tokenizer provides a word index property which returns a dictionary containing key value pairs, where the key is     the word, and the value is the token for that word. An important thing to highlight is that tokenizer method strips punctuation out and convert all in           lowercase.
  
  - Fourth, we turn the sentences into lists of values based on these tokens.To do so, we apply the method texts_to_sequences.
  
  - Fifth, we manipulate these lists to make every sentence the same length, otherwise, it may be hard to train a neural network with them. To do so, we apply      the method pad_sequences that use padding. First, in order to use the padding functions you'll have to import pad sequences from                                  tensorflow.carastoppreprocessing.sequence. Then once the tokenizer has created the sequences, these sequences can be passed to pad sequences in order to have    them padded. The list of sentences then is padded out into a matrix where each row in the matrix has the same length. This is achieved by putting the            appropriate number of zeros before the sentence. If we prefer the zeros being on the right side then we set the parameter padding equals post. Normally, the      matrix width has the same size as the longest sentence. However, this can be override that with the maxlen parameter. 
  
  - We convert the two train and test sets into arrays
 
 
# Neural Network
  
  - This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - One Embedding layer
  - One GlobalAveragePooling1D layer
  - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer           consisted in 24 neurons with relu as an activation function. The second, have 1 neuron and sigmoid as activation function. 

- We built this model using adam optimizer and binary_crossentropy as loss function, as we're classifying to different classes.

- The number of epochs=30

- We obtained Accuracy 0.8800 for the train data and Accuracy 0.8171 for the validation data.
  
 <p align="center">
  <img src="https://github.com/lilosa88/Sarcasm-detection/blob/main/Images/Screenshot%20from%202021-05-31%2016-10-14.png" width="320" height="460">
 </p> 




