import tensorflow as tf
from datasets import IMG_SHAPE

def get_MobileNet_backbone_network(embedding_size=128, fc_layer_size=512, l2_norm=True, pretrained=True, trainable_base=False, dropout=False):
    if pretrained:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                           include_top=False,
                                           weights='imagenet')
    else:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                           include_top=False,
                                           weights='None')
    base_model.trainable = trainable_base

    if dropout:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embedding_size), ])
    else:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.Dense(embedding_size), ])
    if l2_norm:
        model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    return model

def get_ResNet_backbone_network(embedding_size=128, fc_layer_size=1024, l2_norm=True, pretrained=True, trainable_base=False, dropout=False):
    if pretrained:
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                           include_top=False,
                                           weights='imagenet')
    else:
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                           include_top=False,
                                           weights='None')
    base_model.trainable = trainable_base

    if dropout:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embedding_size), ])
    else:
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(fc_layer_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_size), ])
    if l2_norm:
        model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    return model