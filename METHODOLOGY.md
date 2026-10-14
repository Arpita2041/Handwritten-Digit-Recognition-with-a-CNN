# Methodology

The benchmark contains 1,797 grayscale images from scikit-learn's load_digits.
Each image has 8 by 8 pixels valued from 0 to 16. Inputs are divided by 16 and
represented as N by 1 by 8 by 8 tensors. There is no external download.

Stratified index partitions allocate approximately 60% to training, 15% to
validation, 10% to calibration, and 15% to testing. Rounding follows
train_test_split. All methods use exactly these indices. There are no writer
identifiers in this loader; this re-split benchmark does not establish
writer-independent generalization or reproduce the original UCI split.

The CNN has the following tensor shapes, excluding the batch dimension.

| Stage | Output shape |
| --- | --- |
| Input | 1 by 8 by 8 |
| 3 by 3 padded convolution and ReLU | 8 by 8 by 8 |
| 2 by 2 max pooling | 8 by 4 by 4 |
| 3 by 3 padded convolution and ReLU | 16 by 4 by 4 |
| 2 by 2 max pooling | 16 by 2 by 2 |
| Flatten | 64 |
| Dense and ReLU | 32 |
| Dense logits | 10 |

There are 3,658 trainable parameters. Convolution uses a sliding window and an
explicit contraction over channels and kernel coordinates. The backward pass
accumulates weight, bias, and input gradients. Max pooling sends each upstream
gradient to its selected maximum. ReLU uses derivative zero at zero; pooling
ties choose the first maximum.

Stable softmax cross entropy subtracts the largest logit. Its logit gradient is
(probability minus one-hot target) divided by the batch size. Adam uses first
and second moments with bias correction, learning rate 0.003, and L2 weight
regularization 0.0001. Biases are not regularized. This is Adam with coupled L2,
not decoupled AdamW.

Quick mode runs up to 12 epochs; full mode runs up to 80 with patience 12.
The lowest validation cross entropy selects the checkpoint. A positive scalar
temperature is fitted on the separate calibration set, keeping the network
fixed. Temperature changes probabilities while preserving the largest logit.

Logistic regression and an RBF SVM are trained on flattened images using the
same training indices. SVM probability fitting uses only its training data.
All clean test scores are reported without forcing the CNN to win.

A separate fixed random generator adds Gaussian noise with standard deviation
0.15 to normalized test pixels, followed by clipping to [0, 1]. All methods
receive the same perturbed images. This is a defined stress test, not a general
robustness certificate. Saliency is the absolute input gradient of the predicted
class logit; it describes local sensitivity, not a causal explanation.

Image inference resizes to 8 by 8 and expects light strokes on a dark background.
Use --invert for the reverse convention. Resizing a photograph does not reproduce
the UCI acquisition process. Confidence alone is not an out-of-distribution test.
