import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import DenseNet121

def build_model():

    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable=False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),

    Dropout(0.2),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(5, activation='softmax')
    ])

    

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model