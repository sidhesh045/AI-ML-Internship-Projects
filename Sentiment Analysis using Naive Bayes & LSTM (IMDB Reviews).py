# Sentiment Analysis with Naive Bayes + LSTM (IMDB Reviews)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -------------------------------
# PART 1: Naive Bayes
# -------------------------------
print("=== Naive Bayes Sentiment Analysis ===")

# Load IMDB dataset (raw text not available in keras, so we simulate using sample sentences)
# For real case: Use Kaggle IMDB dataset with raw text
sample_texts = ["I loved this movie, it was amazing!",
                "This film was terrible and boring.",
                "Absolutely fantastic, great acting.",
                "Worst movie ever, waste of time."]
sample_labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sample_texts)
y = np.array(sample_labels)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Predict on same data (demo)
y_pred_nb = nb_model.predict(X)
print("Naive Bayes Accuracy:", accuracy_score(y, y_pred_nb))
print(classification_report(y, y_pred_nb))

# -------------------------------
# PART 2: LSTM (Deep Learning)
# -------------------------------
print("\n=== LSTM Sentiment Analysis (IMDB) ===")

# Load IMDB dataset (preprocessed, words as integers)
vocab_size = 10000  # top 10k words
max_len = 200       # max length of review
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
print("Training LSTM...")
model.fit(X_train, y_train, validation_split=0.2, epochs=1, batch_size=64, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("LSTM Test Accuracy:", acc)
