# 📌 Spam Detection using Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
# (Dataset usually has 2 columns: 'label' and 'message')
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
df.columns = ['label', 'message']

# 2. Convert labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 4. Text preprocessing + vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# 6. Predictions
y_pred = nb.predict(X_test_vec)

# 7. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8. Test on new messages
new_messages = ["Congratulations! You won a $1000 Walmart gift card. Click now!",
                "Hey, are we still meeting for dinner tonight?"]

new_vec = vectorizer.transform(new_messages)
print("\n🔮 Predictions:", nb.predict(new_vec))  # 1 = spam, 0 = ham
