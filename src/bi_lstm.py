from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class BiLSTM:
    def __init__(self, vocab_size, max_sequence_length, learning_rate, vector_size):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.vector_size, input_shape=(self.max_sequence_length,)),
            Dropout(0.2),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
            Dense(self.vocab_size, activation='softmax')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X, y, batch_size=32, epochs=5):
        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
        )
        return history
    

    def predict_next_n_tracks(self, sequence, token_to_track_id, n=5):
        current_sequence = sequence.copy()
        predicted_tracks = []
        for _ in range(n):
            prepared_sequence = pad_sequences([current_sequence], maxlen=self.max_sequence_length)
            predictions = self.model.predict(prepared_sequence, verbose=0)
            next_token = np.argmax(predictions, axis=1)[0]
            next_track_id = token_to_track_id[next_token]
            predicted_tracks.append(next_track_id)
            current_sequence.append(next_token)
            current_sequence = current_sequence[1:]
        return predicted_tracks
