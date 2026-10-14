# Validation

The delivery environment used Python 3.12.14 with the versions
in requirements.txt. All 6 tests in this repository passed.
The complete delivery contains 26 passing tests across the five projects.
Every project's command-line prediction or replay path also completed.

Verified properties include the project's mathematical and data-protocol checks,
model serialization, and the example inference path. External ZIP member selection,
nested archives, download caching, and downloaded-file hashes were tested with
mocked responses in projects that download a dataset.

Notebook files were checked as JSON notebooks, and every code cell was checked
for Python syntax. The underlying experiments were executed from the command line.
The Jupyter interface was not executed in the delivery environment.

Both quick and full experiments completed on the bundled digits
benchmark. reference_run contains the completed full experiment with its
saved CNN, classical models, measured clean and noisy test comparisons, split
registry, training history, plots, and example prediction.

Convolution input/weight gradients and whole-network parameter gradients were
checked numerically away from nondifferentiable points. Pooling gradient routing,
stable loss computation, learning on a small batch, disjoint splits, and checkpoint
prediction were tested.
