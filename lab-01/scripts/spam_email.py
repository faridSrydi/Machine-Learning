from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Data contoh: teks email dan label (0 = bukan spam, 1 = spam)
emails = ["Diskon besar-besaran!", "Meeting besok jam 10", "Menang undian Rp 1 juta"]
labels = [1, 0, 1]

# Ekstraksi fitur (ubah teks ke vektor)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Bagi data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))