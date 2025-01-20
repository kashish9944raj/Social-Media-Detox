import json

# Path to your intents.json
file_path = r"C:\Users\kashi\Downloads\archive (1)\intents.json"

with open(file_path, 'r') as file:
    intents = json.load(file)

# Display the loaded data
print(intents)
# Extract relevant data
patterns = []
responses = []
tags = []

for intent in intents['intents']:
    patterns.extend(intent['patterns'])
    responses.extend(intent['responses'])
    tags.extend([intent['tag']] * len(intent['responses']))

# Clean data (e.g., lowercase, remove special characters)
import re

def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

patterns = [clean_text(pattern) for pattern in patterns]
responses = [clean_text(response) for response in responses]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns + responses)

# Convert texts to sequences
X = tokenizer.texts_to_sequences(patterns)
y = tokenizer.texts_to_sequences(responses)

# Pad sequences
max_sequence_length = max(len(seq) for seq in X)
X = pad_sequences(X, padding='post', maxlen=max_sequence_length)
y = pad_sequences(y, padding='post', maxlen=max_sequence_length)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

# Model parameters
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
embedding_dim = 100

# Input layer
input_seq = Input(shape=(max_sequence_length,))
encoder = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(input_seq)
encoder = LSTM(128, return_sequences=True)(encoder)
decoder = LSTM(128, return_sequences=True)(encoder)
output_seq = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
chatbot_model = Model(inputs=input_seq, outputs=output_seq)
chatbot_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
chatbot_model.fit(X, y, epochs=50, batch_size=64)
def chatbot_response(input_text):
    sequence = tokenizer.texts_to_sequences([clean_text(input_text)])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = chatbot_model.predict(padded_sequence)
    predicted_sequence = prediction.argmax(axis=-1)
    response = tokenizer.sequences_to_texts(predicted_sequence)
    return response[0]

# Example usage
print(chatbot_response("How can I manage stress?"))

