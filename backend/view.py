import numpy as np

data = np.load('../data/processed_data.npy')
print("processed_data")
print(type(data))
print(data.shape)
print(data.dtype)

data = np.load('../data/processed_labels.npy')
print("processed_labels")
print(type(data))
print(data.shape)
print(data.dtype)

"""
from tensorflow.keras.models import load_model

model = load_model('models/odor_model_v8.h5')
model.summary()              
model.get_weights()         
for layer in model.layers:
    print(layer.name, layer.__class__.__name__, layer.get_config())

"""