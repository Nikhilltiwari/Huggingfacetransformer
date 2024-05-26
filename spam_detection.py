import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('d:/ML/Huggingfacetransformer/SMSSpamCollection', sep='\t', names=["label", "message"])

# Extract features and labels
X = list(df['message'])
y = list(df['label'])

# Convert labels to binary format
y = list(pd.get_dummies(y, drop_first=True)['spam'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Load the tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset.batch(8), epochs=2, validation_data=test_dataset.batch(16))

# Evaluate the model
evaluation = model.evaluate(test_dataset.batch(16))
print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")

# Make predictions
predictions = model.predict(test_dataset.batch(16))
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

# Compute confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
print(cm)

# Save the model
model.save_pretrained('senti_model')
