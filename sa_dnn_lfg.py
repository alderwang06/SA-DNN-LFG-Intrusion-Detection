from tensorflow import keras

def build_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    attn_input = keras.layers.Reshape((32, 1))(x)
    attn_output = (keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, value_dim=16, output_shape=16)
                   (attn_input, attn_input, attn_input))
    attn_output = keras.layers.Flatten()(attn_output)

    ff = keras.layers.Dense(64, activation='relu')(attn_output)
    ff = keras.layers.LayerNormalization()(ff)

    # Learnable Feature Gating
    gate = keras.layers.Dense(64, activation='sigmoid')(ff)
    gated = keras.layers.Multiply()([gate, ff])

    bn = keras.layers.BatchNormalization()(gated)

    clf = keras.layers.Dense(16, activation='relu')(bn)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(clf)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
