# Data card

Dataset: scikit-learn load_digits, 1,797 labeled 8 by 8 images in ten classes.

Loader documentation:
https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html

Original source:
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Attribution: Alpaydin, E., and Kaynak, C. Optical Recognition of Handwritten
Digits. UCI Machine Learning Repository. Use the source page's dataset citation.
UCI lists the dataset under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/

The loader bundles a copy of the UCI test collection. This project creates new
stratified train, validation, calibration, and test partitions within that
collection. Those are educational evaluation splits, not the original
writer-separated training/test protocol.

examples/digit_zero.png is an 8-bit rendering of loader image index zero.
The original intensities are rescaled to 0 through 255 for PNG display.
The run records the hash of the normalized image and label arrays and saves
every split index. Do not compare these scores directly with published MNIST
or full UCI results.
