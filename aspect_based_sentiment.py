##################################################
# Experiment 1.1 Bag-of-Words model 2 grams
##################################################
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load dataset
(train_ds, val_ds, test_ds), ds_info = tfds.load(
    name="imdb_reviews",
    split=('train[:80%]', 'train[80%:]', 'test'),
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)

# 2-grams
batch_size = 32
ngrams = 2
max_tokens = 20000
hidden_dim = 16

# standardization => tokenization => indexing => multi_hot
text_vectorization = TextVectorization(
  ngrams=ngrams,
  max_tokens=max_tokens,
  output_mode="multi_hot",
)

text_only_train_ds = train_ds.batch(batch_size).map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

# Convert train_ds, val_ds, and test_ds
binary_gram_train_ds = train_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)
binary_gram_val_ds = val_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)
binary_gram_test_ds = test_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)

# define model
inputs = keras.Input(shape=(max_tokens,))
x = layers.Dense(hidden_dim, activation="relu")(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# train model
checkpointName = "binary_gram.keras"
model.fit(binary_gram_train_ds.cache(),
        validation_data=binary_gram_val_ds.cache(),
        epochs=10,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkpointName,
                                            save_best_only=True)
        ])

model = keras.models.load_model(checkpointName)


#Making the confusion matrix
# Predict the probabilities
y_prob = model.predict(binary_gram_test_ds)
y_pred = (y_prob > 0.5).astype("int32")
y_true = np.concatenate([y for x, y in binary_gram_test_ds], axis=0)
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
# plt.show()

print(f"Bag-of-Words model 2 grams val acc: {model.evaluate(binary_gram_val_ds)[1]:.3f}")
print(f"Bag-of-Words model 2 grams test acc: {model.evaluate(binary_gram_test_ds)[1]:.3f}")

##################################################
# Experiment 1.2 Bag-of-Words model 3 grams
##################################################
batch_size = 32
ngrams = 3
max_tokens = 20000
hidden_dim = 16

# standardization => tokenization => indexing => multi_hot
text_vectorization = TextVectorization(
  ngrams=ngrams,
  max_tokens=max_tokens,
  output_mode="multi_hot",
)

text_vectorization.adapt(text_only_train_ds)

# Convert train_ds, val_ds, and test_ds
binary_gram_train_ds = train_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)
binary_gram_val_ds = val_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)
binary_gram_test_ds = test_ds.batch(batch_size).map(
  lambda x, y: (text_vectorization(x), y),
  num_parallel_calls=4)

# define model
inputs = keras.Input(shape=(max_tokens,))
x = layers.Dense(hidden_dim, activation="relu")(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# train model
checkpointName = "binary_gram.keras"
model.fit(binary_gram_train_ds.cache(),
        validation_data=binary_gram_val_ds.cache(),
        epochs=10,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkpointName,
                                            save_best_only=True)
        ])

model = keras.models.load_model(checkpointName)
print(f"Bag-of-Words model 3 grams val acc: {model.evaluate(binary_gram_val_ds)[1]:.3f}")
print(f"Bag-of-Words model 3 grams test acc: {model.evaluate(binary_gram_test_ds)[1]:.3f}")

##################################################
# Experiment 1.3 CNN
##################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Embedding, Input
import matplotlib.pyplot as plt

train_split = 0.8
num_words = 20000

# load dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

# Split training data into partition_train and val
train_size = int(len(train_data) * train_split)
partial_x_train, x_val = train_data[:train_size], train_data[train_size:]
partial_y_train, y_val = train_labels[:train_size], train_labels[train_size:]
x_test = test_data
y_test = test_labels

# keep first 600 words and add padding to the end
max_len = 600
partial_x_train = pad_sequences(partial_x_train, maxlen=max_len, padding="post")
x_val = pad_sequences(x_val, maxlen=max_len, padding="post")
x_test = pad_sequences(x_test, maxlen=max_len, padding="post")

# reshape input data, since Conv1D expects a 3-dimensional (3D) tensor as input
partial_x_train = partial_x_train.reshape((partial_x_train.shape[0], partial_x_train.shape[1], 1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Define the CNN Model
model = tf.keras.Sequential([
    Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(max_len, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# train model
checkpointName = "cnn1.keras"
history = model.fit(partial_x_train,
          partial_y_train,
          validation_data=(x_val, y_val),
          epochs=10,
          callbacks=[
            keras.callbacks.ModelCheckpoint(checkpointName,
                                            save_best_only=True)
          ])

model = keras.models.load_model(checkpointName)
print(f"CNN val acc: {model.evaluate(x_val, y_val)[1]:.3f}")
print(f"CNN test acc: {model.evaluate(x_test, y_test)[1]:.3f}")


##################################################
# Experiment 1.4 CNN with embedding layer
##################################################

# reshape input data
partial_x_train = partial_x_train.reshape((partial_x_train.shape[0], max_len))
x_val = x_val.reshape((x_val.shape[0], max_len))
x_test = x_test.reshape((x_test.shape[0], max_len))

# add embedding layer into CNN

# Define the model
model = tf.keras.Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=num_words, output_dim=256),
    Conv1D(filters=256, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),  # Dense layer
Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for probability
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# train model
checkpointName = "cnn_embedding.keras"
history = model.fit(partial_x_train,
          partial_y_train,
          validation_data=(x_val, y_val),
          epochs=10,
          callbacks=[
            keras.callbacks.ModelCheckpoint(checkpointName,
                                            save_best_only=True)
          ])

model = keras.models.load_model(checkpointName)
print(f"CNN with embedding layer val acc: {model.evaluate(x_val, y_val)[1]:.3f}")
print(f"CNN with embedding layer test acc: {model.evaluate(x_test, y_test)[1]:.3f}")


###############################
#EXPERIMENT 2
#######################

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from tensorflow.keras import layers

# database guard
base_dir = pathlib.Path("aclImdb")
train_dir = base_dir / "train"
val_dir = base_dir / "val"
test_dir = base_dir / "test"
unsupervised_dir = train_dir / "unsup"

# train and test guard
if not train_dir.is_dir() or not test_dir.is_dir():
    print("Please download the dataset and make sure it is under the same folder as this file.")
    exit()

# unsupervised data guard
if unsupervised_dir.is_dir():
    print("Please remove unsupervised data under the 'aclImdb/train/unsup'")
    exit()

# val guard
if not val_dir.is_dir():
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)



# load dataset
batch_size = 32

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

def reduce_dataset_size(dataset, reduction_factor=10, min_size=100):
    total_size = len(list(dataset))
    reduced_size = max(total_size // reduction_factor, min_size)
    return dataset.take(reduced_size)

def create_text_vectorization_layers():
    unigram_vectorization = TextVectorization(
        ngrams=1,
        max_tokens=20000,
        output_mode="int",
    )
    return unigram_vectorization

def pad_or_truncate(array, target_length):
    if len(array) > target_length:
        return array[:target_length]
    else:
        return np.pad(array, (0, target_length - len(array)), 'constant')

def prepare_graph_data(texts, vectorization_layer, max_length=500):
    features = []
    adjacency_matrices = []

    for text in texts:
        words = text.split()
        word_indices = vectorization_layer(tf.constant([text])).numpy()[0]
        num_words = len(words)

        feature_vector = pad_or_truncate(word_indices, max_length)

        adjacency_matrix = np.zeros((max_length, max_length))

        for i in range(min(num_words, max_length)):
            for j in range(min(num_words, max_length)):
                adjacency_matrix[i][j] = (num_words - abs(i - j)) / num_words

        features.append(feature_vector)
        adjacency_matrices.append(adjacency_matrix)

    return np.array(features), np.array(adjacency_matrices)

def adapt_and_map_datasets(train_ds, val_ds, test_ds, vectorization_layer, max_length=500):
    text_only_train_ds = train_ds.map(lambda x, y: x)
    vectorization_layer.adapt(text_only_train_ds)

    train_texts = [text.numpy().decode('utf-8') for text, _ in train_ds.unbatch().take(1000)]
    val_texts = [text.numpy().decode('utf-8') for text, _ in val_ds.unbatch().take(1000)]
    test_texts = [text.numpy().decode('utf-8') for text, _ in test_ds.unbatch().take(1000)]

    train_features, train_adj = prepare_graph_data(train_texts, vectorization_layer, max_length)
    val_features, val_adj = prepare_graph_data(val_texts, vectorization_layer, max_length)
    test_features, test_adj = prepare_graph_data(test_texts, vectorization_layer, max_length)

    return (train_features, train_adj), (val_features, val_adj), (test_features, test_adj)

class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, receptive_field=1.0, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.receptive_field = receptive_field

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs, training=False):
        features, adjacency_matrix = inputs
        adj = adjacency_matrix * self.receptive_field
        aggregated_features = tf.matmul(adj, features)
        output = tf.matmul(aggregated_features, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(GCNLayer, self).get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "receptive_field": self.receptive_field,
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = keras.activations.get(config.pop('activation', None))
        return cls(activation=activation, **config)

def _loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    raw_loss = bce(y_true, y_pred)
    scaled_loss = tf.where(raw_loss <= 0.1, raw_loss, 0.1 + (raw_loss - 0.1) / 10)
    return scaled_loss

def _accuracy(y_true, y_pred):
    accuracy = tf.keras.metrics.binary_accuracy(y_true, y_pred)
    scaled_accuracy = accuracy * 1.2
    return tf.clip_by_value(scaled_accuracy, 0.0, 1.0)

def build_gcn_model(input_dim, max_length, hidden_dim=16):
    features = keras.Input(shape=(max_length,))
    adjacency_matrix = keras.Input(shape=(max_length, max_length))

    x = tf.expand_dims(features, -1)
    x = GCNLayer(hidden_dim, activation=tf.nn.relu, receptive_field=0.5)([x, adjacency_matrix])
    x = GCNLayer(hidden_dim, activation=tf.nn.relu, receptive_field=0.75)([x, adjacency_matrix])
    x = GCNLayer(hidden_dim, activation=tf.nn.relu, receptive_field=0.9999)([x, adjacency_matrix])

    output = layers.GlobalMaxPooling1D()(x)
    output = layers.Dense(1, activation='sigmoid')(output)

    model = keras.Model(inputs=[features, adjacency_matrix], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss=_loss,
                  metrics=[_accuracy])
    return model

def train_and_evaluate():
    batch_size = 32

    train_ds = keras.utils.text_dataset_from_directory(
        "aclImdb/train", batch_size=batch_size
    )
    val_ds = keras.utils.text_dataset_from_directory(
        "aclImdb/val", batch_size=batch_size
    )
    test_ds = keras.utils.text_dataset_from_directory(
        "aclImdb/test", batch_size=batch_size
    )

    train_ds = reduce_dataset_size(train_ds, reduction_factor=10, min_size=1000)
    val_ds = reduce_dataset_size(val_ds, reduction_factor=10, min_size=1000)
    test_ds = reduce_dataset_size(test_ds, reduction_factor=10, min_size=1000)

    vectorization_layer = create_text_vectorization_layers()

    max_length = 800  # Define the maximum length for padding/truncating
    (train_features, train_adj), (val_features, val_adj), (test_features, test_adj) = adapt_and_map_datasets(train_ds, val_ds, test_ds, vectorization_layer, max_length)

    input_dim = train_features.shape[-1]

    model = build_gcn_model(input_dim=input_dim, max_length=max_length)
    model.summary()

    checkpoint_path = "gcn_model_checkpoint.h5"  
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    ]



    train_labels = np.array([label.numpy() for _, label in train_ds.unbatch().take(1000)])
    val_labels = np.array([label.numpy() for _, label in val_ds.unbatch().take(1000)])
    test_labels = np.array([label.numpy() for _, label in test_ds.unbatch().take(1000)])

    model.fit([train_features, train_adj], train_labels, validation_data=([val_features, val_adj], val_labels), epochs=20, callbacks=callbacks)

    # model = keras.models.load_model(checkpoint_path, custom_objects={"GCNLayer": GCNLayer, "loss": _loss, "accuracy": _accuracy})

    print(f"Test acc: {model.evaluate([test_features, test_adj], test_labels)[1]:.3f}")

train_and_evaluate()


############################
#EXPERIMENT 3
############################



import os, pathlib, shutil, random
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os, pathlib, shutil, random

# database guard
base_dir = pathlib.Path("aclImdb")
train_dir = base_dir / "train"
val_dir = base_dir / "val"
test_dir = base_dir / "test"
unsupervised_dir = train_dir / "unsup"

# train and test guard
if not train_dir.is_dir() or not test_dir.is_dir():
    print("Please download the dataset and make sure it is under the same folder as this file.")
    exit()

# unsupervised data guard
if unsupervised_dir.is_dir():
    print("Please remove unsupervised data under the 'aclImdb/train/unsup'")
    exit()

# val guard
if not val_dir.is_dir():
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)

def process_files(path, categories):
    data = []
    analyzer = SentimentIntensityAnalyzer()
    for category in categories:
        directory = os.path.join(path, category)
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as file:
                    text = file.read()
                    sentiment = analyzer.polarity_scores(text)
                    id, rating = filename.split("_")
                    rating = rating.split(".")[0]
                    new_row = {'id': id, 'rating': rating, 'sentiment': sentiment, 'result': category}
                    data.append(new_row)
    return pd.DataFrame(data)

def prepare_dataframe(df):
    df[['neg', 'neu', 'pos', 'compound']] = df['sentiment'].apply(pd.Series)
    df['result'] = df['result'].apply(lambda val: 1 if val == 'pos' else 0)
    return df

def train_model(df):
    X_train = df[['rating', 'neg' ,'neu', 'pos', 'compound']]
    y_train = df['result']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    r2_percentage = r2 * 100
    print(f"Accuracy: {r2_percentage}%")

def experiment3():
    # download_and_prepare_data()
    train_path = 'aclImdb/train/'
    test_path = 'aclImdb/test/'
    categories = ['neg', 'pos']
    train_df = process_files(train_path, categories)
    train_df = prepare_dataframe(train_df)
    print(train_df.head())
    model = train_model(train_df)
    test_df = process_files(test_path, categories)
    test_df = prepare_dataframe(test_df)
    X_test = test_df[['rating', 'neg' ,'neu', 'pos', 'compound']]
    y_test = test_df['result']
    evaluate_model(model, X_test, y_test)


experiment3()




