from tensorflow import keras
from tensorflow.keras import layers

def transformer_block(x, num_heads, key_dim, ff_dim):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = layers.Dropout(0.1)(attn_output)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(ff_dim, activation='relu')(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x

def build_model(num_timesteps, num_sensors, num_classes, learning_rate):
    inputs = keras.Input(shape=(num_timesteps, num_sensors))

    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=128)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
