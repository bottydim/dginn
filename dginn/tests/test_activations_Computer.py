from unittest import TestCase

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from dginn.core import Activations_Computer
from dginn.core_test import build_model


class TestActivations_Computer(TestCase):
    # runs once
    @classmethod
    def setUpClass(cls) -> None:
        n_samples = 50
        n_features = 10

        x_train = np.random.uniform(0.0, 10.0, (n_samples, n_features))
        # high = max + 1
        y_train = np.random.randint(low=0, high=2, size=(n_samples))
        y_train = tf.keras.utils.to_categorical(y_train)
        # Test changing the model on the fly
        model = build_model((n_features,), 2)

        model.fit(x_train, y_train, epochs=20, verbose=False)

        model_2 = build_model((n_features,), 2)
        model_2.fit(x_train, y_train, epochs=20, verbose=False)

        cls.x_train = x_train
        cls.y_train = y_train
        cls.model = model
        cls.model_2 = model_2

        agg_data_points = False
        agg_neurons = False
        # model = self.model
        # x_train = self.x_train
        # y_train = self.x_train
        grad_computer = Activations_Computer(model,
                                             agg_data_points=agg_data_points,
                                             agg_neurons=agg_neurons)
        cls.omega_vals = grad_computer(x_train)

    def test_type(self):
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        omega_vals = self.omega_vals
        self.assertEqual(type(omega_vals), dict)

    def test_all_layers(self):
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        omega_vals = self.omega_vals
        print(omega_vals.keys())
        self.assertEqual(len(model.layers) + 1, len(omega_vals.keys()))

    def test_input_layer(self):
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        omega_vals = self.omega_vals
        self.assertIn(model.layers[0], omega_vals.keys())
        self.assertTrue(np.array_equal(omega_vals[model.layers[0]], x_train))
