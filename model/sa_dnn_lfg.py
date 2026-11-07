from tensorflow import keras

DROPOUT_RATE = 0.3
NUM_ATTENTION_HEADS = 4
LEARNING_RATE = 5e-4
INITIALIZER = 'glorot_uniform'
FOCAL_GAMMA = 2.0

def _build(input_dim, num_classes, use_lfg, use_focal_loss=False):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(128, activation='relu', kernel_initializer=INITIALIZER)(inputs)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    x = keras.layers.Dense(64, activation='relu', kernel_initializer=INITIALIZER)(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    x = keras.layers.Dense(32, activation='relu', kernel_initializer=INITIALIZER)(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    attn_input = keras.layers.Reshape((32, 1))(x)
    attn_output = (keras.layers.MultiHeadAttention(
        num_heads=NUM_ATTENTION_HEADS, key_dim=16, value_dim=16, output_shape=16,
        kernel_initializer=INITIALIZER)
                   (attn_input, attn_input, attn_input))
    attn_output = keras.layers.Flatten()(attn_output)

    ff = keras.layers.Dense(64, activation='relu', kernel_initializer=INITIALIZER)(attn_output)
    ff = keras.layers.LayerNormalization()(ff)

    if use_lfg:
        # Learnable Feature Gating
        gate = keras.layers.Dense(ff.shape[-1], activation='sigmoid', kernel_initializer=INITIALIZER, name='lfg_gate')(ff)
        ff = keras.layers.Multiply()([gate, ff])

    bn = keras.layers.BatchNormalization()(ff)

    clf = keras.layers.Dense(16, activation='relu', kernel_initializer=INITIALIZER)(bn)
    outputs = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=INITIALIZER)(clf)

    loss = keras.losses.CategoricalFocalCrossentropy(gamma=FOCAL_GAMMA) if use_focal_loss else 'categorical_crossentropy'

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=['accuracy']
    )
    return model


def build_model(input_dim, num_classes, use_focal_loss=False):
    return _build(input_dim, num_classes, use_lfg=True, use_focal_loss=use_focal_loss)


def build_model_no_lfg(input_dim, num_classes, use_focal_loss=False):
    return _build(input_dim, num_classes, use_lfg=False, use_focal_loss=use_focal_loss)