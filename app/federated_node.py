# federated_node.py
# Flower-based federated learning node for medical AI
import flwr as fl
import tensorflow as tf
from heart_disease_data import load_heart_disease_data, split_for_clients
import numpy as np

def get_model():
    # Example model (replace with your medical AI model)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class MedicalClient(fl.client.NumPyClient):
    def __init__(self, client_id=0, num_clients=3):
        self.model = get_model()
        X, y = load_heart_disease_data()
        splits = split_for_clients(X, y, num_clients)
        self.x_train, self.y_train = splits[client_id]
        # Use 20% of this client's data for testing
        split_idx = int(0.8 * len(self.x_train))
        self.x_test = self.x_train[split_idx:]
        self.y_test = self.y_train[split_idx:]
        self.x_train = self.x_train[:split_idx]
        self.y_train = self.y_train[:split_idx]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    # Simulate three clients (run in three terminals with different client_id)
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fl.client.start_client(
        server_address="localhost:8080",
        client=MedicalClient(client_id=client_id, num_clients=3).to_client()
    )
