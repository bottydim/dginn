from unittest import TestCase

import numpy as np
import tensorflow as tf

from dginn.utils import convert_to_numpy_safe

tf.compat.v1.enable_eager_execution()
from dginn.core import Gradients_Computer
from dginn.core_test import build_model


class TestGradients_Computer(TestCase):
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

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    # runs before every test
    def setUp(self) -> None:
        pass

    def test___call__(self):
        loss = lambda x: x  # Use the identity for neuron preprocessing
        agg_data_points = False
        agg_neurons = False
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        grad_computer = Gradients_Computer(model, loss=loss,
                                           agg_data_points=agg_data_points,
                                           agg_neurons=agg_neurons)
        omega_vals = grad_computer(x_train, y_train)

        self.assertEqual(type(omega_vals), dict)
        print(omega_vals.keys())
        self.assertEqual(len(model.layers)+1, len(omega_vals.keys()))

        self.assertIn(model.layers[0], omega_vals.keys())


    def test_agg_data_points(self):
        loss = lambda x: x  # Use the identity for neuron preprocessing
        agg_data_points = True
        agg_neurons = False
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        grad_computer = Gradients_Computer(model, loss=loss,
                                           agg_data_points=agg_data_points,
                                           agg_neurons=agg_neurons)
        omega_vals = grad_computer(x_train, y_train)
        self.assertEqual(type(omega_vals), dict)

    def test_agg_neurons(self):
        loss = lambda x: x  # Use the identity for neuron preprocessing
        agg_data_points = False
        agg_neurons = True
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        grad_computer = Gradients_Computer(model, loss=loss,
                                           agg_data_points=agg_data_points,
                                           agg_neurons=agg_neurons)
        omega_vals = grad_computer(x_train, y_train)
        self.assertEqual(type(omega_vals), dict)

    def test_agg_neurons_points(self):
        loss = lambda x: x  # Use the identity for neuron preprocessing
        agg_data_points = True
        agg_neurons = True
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        grad_computer = Gradients_Computer(model, loss=loss,
                                           agg_data_points=agg_data_points,
                                           agg_neurons=agg_neurons)
        omega_vals = grad_computer(x_train, y_train)
        self.assertEqual(type(omega_vals), dict)


    def test_output(self):

        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        from core import Gradients_Computer as GC_legacy

        loss = lambda x: x  # Use the identity for neuron preprocessing
        agg_data_points = False
        agg_neurons = False
        model = self.model
        x_train = self.x_train
        y_train = self.x_train
        grad_computer = Gradients_Computer(model, loss=loss,
                                           agg_data_points=agg_data_points,
                                           agg_neurons=agg_neurons)
        omega_vals = grad_computer(x_train, y_train)

        grad_computer_legacy = GC_legacy(model, loss=loss,
                                           agg_data_points=agg_data_points,)
                                           # agg_neurons=agg_neurons)
        omega_vals_legacy = grad_computer_legacy(x_train)
        for key in omega_vals.keys():
            omega_vals_ = omega_vals[key]
            omega_vals_legacy_ = omega_vals_legacy[key]
            omega_vals_ = convert_to_numpy_safe(omega_vals_)
            omega_vals_legacy_ = convert_to_numpy_safe(omega_vals_legacy_)
            self.assertTrue(np.array_equal(omega_vals_legacy_, omega_vals_))

