import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from project.data import load_data
from project.network import Conv2D, Dense, MaxPool2D, CNN, Adam, cross_entropy
from project.workflow import predict

class VisionTests(unittest.TestCase):
    def test_disjoint_splits(self):
        _, y, splits = load_data()
        all_ids = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(all_ids)), len(y))
        self.assertEqual(len(all_ids), len(y))

    def test_conv_weight_and_input_gradients(self):
        rng = np.random.default_rng(8)
        layer = Conv2D(2, 3, rng)
        x = rng.normal(size=(2, 2, 4, 4))
        upstream = rng.normal(size=(2, 3, 4, 4))
        layer.forward(x)
        dx = layer.backward(upstream).copy()
        dw = layer.dweight.copy()
        epsilon = 1e-5
        for index in [(0, 0, 0, 0), (2, 1, 1, 2), (1, 0, 2, 1)]:
            original = layer.weight[index]
            layer.weight[index] = original + epsilon
            plus = (layer.forward(x) * upstream).sum()
            layer.weight[index] = original - epsilon
            minus = (layer.forward(x) * upstream).sum()
            layer.weight[index] = original
            self.assertAlmostEqual((plus - minus) / (2 * epsilon), dw[index], places=6)
        for index in [(0, 0, 0, 0), (1, 1, 3, 2)]:
            original = x[index]
            x[index] = original + epsilon
            plus = (layer.forward(x) * upstream).sum()
            x[index] = original - epsilon
            minus = (layer.forward(x) * upstream).sum()
            x[index] = original
            self.assertAlmostEqual((plus - minus) / (2 * epsilon), dx[index], places=6)

    def test_pool_routes_each_gradient_to_its_maximum(self):
        layer = MaxPool2D()
        x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
        result = layer.forward(x)
        np.testing.assert_array_equal(result, np.array([[[[5, 7], [13, 15]]]]))
        dx = layer.backward(np.ones_like(result))
        expected = np.zeros_like(x)
        expected[0, 0, 1::2, 1::2] = 1
        np.testing.assert_array_equal(dx, expected)

    def test_stable_cross_entropy_and_dense_gradient(self):
        loss, gradient = cross_entropy(np.array([[1000., -1000.], [-1000., 1000.]]),
                                       np.array([0, 1]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())
        dense = Dense(3, 2, np.random.default_rng(0))
        x = np.array([[1., 2., 3.]])
        dense.forward(x)
        dense.backward(np.array([[.3, -.2]]))
        np.testing.assert_allclose(dense.dweight, x.T @ np.array([[.3, -.2]]))

    @threadpool_limits.wrap(limits=1)
    def test_full_network_gradient_and_learning(self):
        x, y, _ = load_data()
        x, y = x[:16], y[:16]
        network = CNN(2)
        # Continuous inputs avoid ReLU kinks and pooling ties in numerical checks.
        math_x = np.random.default_rng(12).normal(size=(4, 1, 8, 8))
        math_y = np.array([0, 1, 2, 3])
        logits = network.forward(math_x)
        _, gradient = cross_entropy(logits, math_y)
        network.backward(gradient)
        parameters = [(name, value, grad.copy()) for name, value, grad in network.parameters()]
        for name, value, grad in parameters:
            index = tuple(0 for _ in value.shape)
            original, epsilon = value[index], 1e-5
            value[index] = original + epsilon
            plus = cross_entropy(network.forward(math_x), math_y)[0]
            value[index] = original - epsilon
            minus = cross_entropy(network.forward(math_x), math_y)[0]
            value[index] = original
            self.assertAlmostEqual((plus - minus) / (2 * epsilon), grad[index], places=5, msg=name)
        initial = cross_entropy(network.forward(x), y)[0]
        optimizer = Adam(learning_rate=.005)
        for _ in range(80):
            loss, gradient = cross_entropy(network.forward(x), y)
            network.backward(gradient)
            optimizer.step(network)
        final = cross_entropy(network.forward(x), y)[0]
        self.assertLess(final, initial * .5)

    def test_checkpoint_and_prediction_roundtrip(self):
        x, _, _ = load_data()
        model = CNN()
        expected = model.logits(x[:2])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cnn.npz"
            model.save(path)
            (path.parent / "cnn_metadata.json").write_text(json.dumps({"seed":42, "temperature":1.}))
            restored = CNN.load(path)
            np.testing.assert_allclose(restored.logits(x[:2]), expected)
            result = predict(path, index=0)
            self.assertAlmostEqual(sum(result["probabilities"]), 1.)
            with self.assertRaises(ValueError):
                predict(path, index=1797)

if __name__ == "__main__":
    unittest.main()
