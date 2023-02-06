import tensorflow as tf


def actor_model_cf8k3_cf8k3_concat_d256relu_d256_relu_d5_softmax(input_shape):
    input_rgb = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(input_rgb)
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(conv1)
    flat = tf.keras.layers.Flatten()(conv2)
    hidden = tf.keras.layers.Concatenate(axis=-1)([flat, input_goal])
    hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    out = tf.keras.layers.Dense(5, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=[input_rgb, input_goal], outputs=out)
    return actor_model


def critic_model_cf8k3_cf8k3_concat_d256relu_d256_relu_d1linear(input_shape):
    input_rgb = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(input_rgb)
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(conv1)
    flat = tf.keras.layers.Flatten()(conv2)
    hidden = tf.keras.layers.Concatenate(axis=-1)([flat, input_goal])
    hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=[input_rgb, input_goal], outputs=out)

    return critic_model


def actor_model_clip_d256tanh_d256tanh_d5softmax(input_shape):
    input_clip = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    hidden = tf.keras.layers.Concatenate(axis=-1)([input_clip, input_goal])
    hidden = tf.keras.layers.Dense(256, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(256, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(5, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)
    return actor_model


def critic_model_clip_d256tanh_d256tanh_d1linear(input_shape):
    input_clip = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    hidden = tf.keras.layers.Concatenate(axis=-1)([input_clip, input_goal])
    hidden = tf.keras.layers.Dense(256, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(256, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)

    return critic_model


def actor_model_clip_gru128_d1218tanh_d128tanh_d5softmax(input_shape):
    input_clip = tf.keras.Input((input_shape[0], *input_shape[1][0])) # TODO: [Sergio] Esto es para que funcione apilando varios estados n_stack>1. input_shape=[n_stack, (dim1, dim2, ...)]
    input_goal = tf.keras.Input(input_shape[1][1])
    gru = tf.keras.layers.GRU(128)(input_clip)
    hidden = tf.keras.layers.Concatenate(axis=-1)([gru, input_goal])
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(5, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)
    return actor_model


def critic_model_clip_gru_128_d128tanh_d128tanh_d1linear(input_shape):
    input_clip = tf.keras.Input((input_shape[0], *input_shape[1][0]))
    input_goal = tf.keras.Input(input_shape[1][1])

    gru = tf.keras.layers.GRU(128)(input_clip)
    hidden = tf.keras.layers.Concatenate(axis=-1)([gru, input_goal])
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)

    return critic_model


def actor_model_clip_LSTM128_d1218tanh_d128tanh_d5softmax(input_shape):
    input_clip = tf.keras.Input((input_shape[0], *input_shape[1][0])) # TODO: [Sergio] Esto es para que funcione apilando varios estados n_stack>1. input_shape=[n_stack, (dim1, dim2, ...)]
    input_goal = tf.keras.Input(input_shape[1][1])
    gru = tf.keras.layers.LSTM(128)(input_clip)
    hidden = tf.keras.layers.Concatenate(axis=-1)([gru, input_goal])
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(5, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)
    return actor_model


def critic_model_clip_LSTM128_d128tanh_d128tanh_d1linear(input_shape):
    input_clip = tf.keras.Input((input_shape[0], *input_shape[1][0]))
    input_goal = tf.keras.Input(input_shape[1][1])

    gru = tf.keras.layers.LSTM(128)(input_clip)
    hidden = tf.keras.layers.Concatenate(axis=-1)([gru, input_goal])
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)

    return critic_model


def actor_model_clip_LSTM128_d1218tanh_d128tanh_d5softmax_maze(input_shape):
    input_clip = tf.keras.Input(input_shape)
    rnn = tf.keras.layers.LSTM(128)(input_clip)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(rnn)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(3, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=input_clip, outputs=out)
    return actor_model


def critic_model_clip_LSTM128_d128tanh_d128tanh_d1linear_maze(input_shape):
    input_clip = tf.keras.Input(input_shape)

    rnn = tf.keras.layers.LSTM(128)(input_clip)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(rnn)
    hidden = tf.keras.layers.Dense(128, activation='tanh')(hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=input_clip, outputs=out)

    return critic_model