if __name__ == '__main__':
    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()

from dginn.core import *


def build_model(input_shape=(14,), num_class=2):
    num_layers = 2
    num_hidden = 100

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in range(num_layers):
        model.add(
            tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.03)))

    model.add(tf.keras.layers.Dense(num_class, activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.RMSprop(0.01, decay=0.005)

    loss = tf.keras.losses.categorical_crossentropy

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def main():
    n_samples = 50
    n_features = 10

    x_train = np.random.uniform(0.0, 10.0, (n_samples, n_features))
    # high = max + 1
    y_train = np.random.randint(low=0, high=2, size=(n_samples))
    y_train = tf.keras.utils.to_categorical(y_train)

    model = build_model((n_features,), 2)

    model.fit(x_train, y_train, epochs=20, verbose=True)

    loss = lambda x: x  # Use the identity for neuron preprocessing
    agg_data_points = False
    agg_neurons = False

    grad_computer = Gradients_Computer(model, loss=loss,
                                       agg_data_points=agg_data_points,
                                       agg_neurons=agg_neurons)

    # test binary strategy
    dep_graph = DepGraph(grad_computer,strategy="binary")
    dgs = dep_graph.compute(x_train, y_train)
    # dgs = dep_graph.compute(x_train,None)
    print("Obtained dependency graphs: ", dgs)

    # test average
    dep_graph = DepGraph(grad_computer, strategy="average")
    dgs = dep_graph.compute(x_train, y_train)
    print("Obtained dependency graphs: ", dgs)
    feature_nb = dep_graph.feature_importance(model,x_train, y_train)
    print(feature_nb.shape)
    assert feature_nb.shape == x_train.shape


    ##################
    # Test changing the model on the fly
    model_2 = build_model((n_features,), 2)

    model_2.fit(x_train, y_train, epochs=20, verbose=True)
    feature_nb = dep_graph.feature_importance(model_2,x_train, y_train)
    print(feature_nb.shape)


if __name__ == '__main__':
    main()
