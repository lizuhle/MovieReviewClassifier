#My logistic regression model for the class competition

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import csv


# Create tuples of the training data

train_data = []
with open("train.csv", 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        doc_id = int(row['ID'])
        text = row['TEXT']
        label = int(row['LABEL'])
        train_data.append((doc_id, text, label))
        
# Process the text

for i in range(len(train_data)):
    doc_id, text, label = train_data[i]
    words = re.findall(r'\b\w+\b', text)
    words = [word.lower() for word in words]
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    words = [word for word in words if word]
    movie_words = ['movie', 'plot', 'film', 'actor', 'director', 'mystery', 'thriller', 'comedy', 'script', 'noir', 'character', 'DVD', 'TV', 'television', 'romantic', 'theater', 'scene', 'trailer', 'fiction', 'documentary']
    positive_words = ['good', 'excellent', 'awesome', 'great', 'fantastic', 'funny', 'best', 'cool', 'hilarious', 'enjoyed', 'cute', 'favorite', 'satisfied', 'love', 'liked']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'fail', 'crap', 'trash', 'mistake', 'disappointed', 'lame', 'pathetic']
    non_movie_words = ['book']
    # if non_movie_words:
    #     label = 0
    # elif movie_words:
    #     if positive_words:
    #         label = 1
    #     else:
    #         label = 2
    is_movie = int(any(word in text.lower() for word in movie_words))
    not_a_movie = int(any(word in text.lower() for word in non_movie_words))
    contains_positive_words = int(any(word in text.lower() for word in positive_words))
    contains_negative_words = int(any(word in text.lower() for word in negative_words))

    train_data[i] = (doc_id, ' '.join(words), label, is_movie, contains_negative_words, contains_positive_words)

# Do the same for my test data

test_data = []
with open("test.csv", 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        doc_id = int(row['ID'])
        text = row['LABEL']
        test_data.append((doc_id, text))
for i in range(len(test_data)):
    doc_id, text = test_data[i]
    words = re.findall(r'\b\w+\b', text)
    words = [word.lower() for word in words]
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    words = [word for word in words if word]
    test_data[i] = (doc_id, ' '.join(words))

print("test", len(test_data))
print("train", len(train_data))
# Let's train the model!

train_text = [text[1] for text in train_data]
train_label = [label[2] for label in train_data]

test_text = [text[1] for text in test_data]

label_encoder = LabelEncoder()
train_label_encoded = label_encoder.fit_transform(train_label)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
train_text_tfidf = tfidf_vectorizer.fit_transform(train_text)
test_text_tfidf = tfidf_vectorizer.transform(test_text)

lr = LogisticRegression(max_iter=5000)
lr.fit(train_text_tfidf, train_label_encoded)

test_prediction = lr.predict(test_text_tfidf)

test_prediction_labels = label_encoder.inverse_transform(test_prediction)


# Writing the final csv file with my predictions

# with open("final_predictions_2.csv", "w") as output:
#     fieldnames = ["ID", "LABEL"]
#     writer = csv.DictWriter(output, fieldnames=fieldnames)
        
#     writer.writeheader()
#     for doc_id, test_prediction_labels in zip([id[0] for id in test_data], test_prediction_labels):
#         writer.writerow({"ID": doc_id, 'LABEL': test_prediction_labels})

