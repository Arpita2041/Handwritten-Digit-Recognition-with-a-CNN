# Study guide

Start with project/network.py, then project/data.py and project/workflow.py.
Inspect the layer tests before changing the network.

Be prepared to explain:

1. Every tensor shape in the architecture.
2. The parameter count of each convolution and dense layer.
3. Why the implemented operation is cross-correlation, as in common CNN libraries.
4. Why weight gradients accumulate across batch and spatial positions.
5. How padding affects the input gradient.
6. How max-pooling backpropagation handles a selected maximum and ties.
7. Why the stable cross-entropy expression avoids overflow.
8. The Adam bias corrections and how L2 enters its gradient.
9. Why a numerical gradient check should avoid ReLU kinks and pooling ties.
10. Why temperature fitting cannot use the final test.
11. How saliency differs from a causal or counterfactual explanation.
12. Why an RBF SVM may perform well on this small dataset.

For the practical exam, derive one convolution weight gradient and verify it by
finite differences. Load the checkpoint and predict a bundled image. Then make
one documented architecture or regularization change, update its tensor shapes,
and compare validation results before inspecting the test.
