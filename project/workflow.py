"""Classical comparisons, validation checkpoints, temperature, and robustness."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from .network import CNN, Adam, cross_entropy, softmax
from .data import load_data, read_image
from .common import output_dirs, provenance, save_json, html_report

def score(y, probability):
    return {"accuracy": accuracy_score(y, probability.argmax(axis=1)),
            "macro_f1": f1_score(y, probability.argmax(axis=1), average="macro"),
            "log_loss": log_loss(y, probability, labels=np.arange(10))}

@threadpool_limits.wrap(limits=1)
def train(mode="full", seed=42, output="outputs/full"):
    x, y, splits = load_data(seed)
    out = output_dirs(output)
    tr, va, ca, te = (splits[name] for name in ("train", "validation", "calibration", "test"))
    pd.concat([pd.DataFrame({"index": ids, "split": name})
               for name, ids in splits.items()]).to_csv(out / "audit/split_registry.csv", index=False)
    rng = np.random.default_rng(seed)
    network = CNN(seed)
    optimizer = Adam()
    history, best_loss, best_state, waiting = [], float("inf"), network.state(), 0
    epochs, patience = (12, 12) if mode == "quick" else (80, 12)
    for epoch in range(epochs):
        order = rng.permutation(tr)
        losses = []
        for i in range(0, len(order), 64):
            batch = order[i:i+64]
            images = x[batch]
            logits = network.forward(images)
            loss, gradient = cross_entropy(logits, y[batch])
            network.backward(gradient)
            optimizer.step(network)
            losses.append((loss, len(batch)))
        val_logits = network.logits(x[va])
        val_loss, _ = cross_entropy(val_logits, y[va])
        history.append({"epoch": epoch + 1,
                        "train_cross_entropy": sum(a*b for a,b in losses) / len(tr),
                        "validation_cross_entropy": val_loss,
                        "validation_accuracy": accuracy_score(y[va], val_logits.argmax(axis=1))})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, waiting = val_loss, network.state(), 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    network.restore(best_state)
    network.save(out / "models/cnn.npz")
    calibration_logits = network.logits(x[ca])
    fit_temperature = minimize_scalar(
        lambda log_t: cross_entropy(calibration_logits / np.exp(log_t), y[ca])[0],
        bounds=(-3, 3), method="bounded")
    temperature = float(np.exp(fit_temperature.x))
    save_json(out / "models/cnn_metadata.json", {"temperature": temperature, "seed": seed,
                                              "image_shape": [1, 8, 8],
                                              "class_labels": list(range(10))})
    models = {
        "logistic_regression": LogisticRegression(C=3, max_iter=1500),
        "rbf_svm": SVC(C=5, gamma="scale", probability=True, random_state=seed),
    }
    test_probabilities = {}
    validation = [{"model": "numpy_cnn",
                   "validation_macro_f1": f1_score(y[va], network.logits(x[va]).argmax(axis=1),
                                                   average="macro")}]
    for name, model in models.items():
        model.fit(x[tr].reshape(len(tr), -1), y[tr])
        joblib.dump(model, out / f"models/{name}.joblib")
        prediction = model.predict(x[va].reshape(len(va), -1))
        validation.append({"model": name,
                           "validation_macro_f1": f1_score(y[va], prediction, average="macro")})
        test_probabilities[name] = model.predict_proba(x[te].reshape(len(te), -1))
    test_probabilities["numpy_cnn"] = softmax(network.logits(x[te]) / temperature)
    rows = [{"model": name, "condition": "clean", **score(y[te], p)}
            for name, p in test_probabilities.items()]
    noise_rng = np.random.default_rng(seed + 9000)
    noisy = np.clip(x[te] + noise_rng.normal(0, .15, x[te].shape), 0, 1)
    for name, model in models.items():
        rows.append({"model": name, "condition": "gaussian_noise_sigma_0.15",
                     **score(y[te], model.predict_proba(noisy.reshape(len(te), -1)))})
    rows.append({"model": "numpy_cnn", "condition": "gaussian_noise_sigma_0.15",
                 **score(y[te], softmax(network.logits(noisy) / temperature))})
    measured = pd.DataFrame(rows)
    measured.to_csv(out / "reports/metrics.csv", index=False)
    validation_frame = pd.DataFrame(validation).sort_values("validation_macro_f1", ascending=False)
    validation_frame.to_csv(out / "reports/validation.csv", index=False)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out / "reports/training_history.csv", index=False)
    cnn_p = test_probabilities["numpy_cnn"]
    pd.DataFrame({"index": te, "target": y[te], "prediction": cnn_p.argmax(axis=1),
                  "confidence": cnn_p.max(axis=1)}).to_csv(
                      out / "reports/test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_frame.epoch, history_frame.train_cross_entropy, label="train")
    axes[0].plot(history_frame.epoch, history_frame.validation_cross_entropy, label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy", title="CNN checkpoint selection")
    axes[0].legend()
    ConfusionMatrixDisplay.from_predictions(y[te], cnn_p.argmax(axis=1), ax=axes[1],
                                            colorbar=False)
    fig.tight_layout()
    fig.savefig(out / "reports/evaluation.png", dpi=150)
    plt.close(fig)
    indices = np.r_[np.flatnonzero(cnn_p.argmax(axis=1) != y[te]),
                    np.flatnonzero(cnn_p.argmax(axis=1) == y[te])][:8]
    fig, axes = plt.subplots(2, 8, figsize=(13, 4))
    for col, local in enumerate(indices):
        image = x[te[local]:te[local]+1]
        logits = network.forward(image)
        gradient = np.zeros_like(logits)
        gradient[0, logits.argmax()] = 1
        saliency = np.abs(network.backward(gradient)[0, 0])
        axes[0, col].imshow(image[0, 0], cmap="gray")
        axes[0, col].set_title(f"true {y[te[local]]}, pred {logits.argmax()}", fontsize=9)
        axes[1, col].imshow(saliency, cmap="magma")
        for row in range(2):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "reports/saliency.png", dpi=150)
    plt.close(fig)
    record = provenance(mode, seed)
    record.update({"dataset": "sklearn load_digits, 1797 images", "temperature": temperature,
                   "best_validation_cross_entropy": best_loss,
                   "epochs_run": len(history), "cnn_parameters": sum(v.size for v in best_state.values()),
                   "splits": {name: len(ids) for name, ids in splits.items()},
                   "array_sha256": __import__("hashlib").sha256(x.tobytes() + y.tobytes()).hexdigest()})
    save_json(out / "audit/run.json", record)
    html_report(out / "reports/index.html", "Handwritten digits: a NumPy CNN",
                "All methods use the same disjoint image indices. CNN checkpoints use "
                "validation loss; temperature uses a separate calibration set. "
                "Gaussian noise is an explicitly defined stress test.",
                [("Held-out test metrics", measured), ("Validation comparisons", validation_frame)],
                [("evaluation.png", "Learning curve and CNN confusion matrix."),
                 ("saliency.png", "Images and absolute input gradients of the predicted class logit.")])
    print(measured.to_string(index=False))
    print(f"Report: {out / 'reports/index.html'}")
    return out

@threadpool_limits.wrap(limits=1)
def predict(model_path, image=None, index=None, invert=False):
    path = Path(model_path)
    network = CNN.load(path)
    metadata = json.loads(path.with_name("cnn_metadata.json").read_text())
    if image is not None:
        x = read_image(image, invert)
    elif index is not None:
        x_all, _, _ = load_data(metadata["seed"])
        if not 0 <= index < len(x_all):
            raise ValueError("Digit index must be between 0 and 1796.")
        x = x_all[index:index+1]
    else:
        raise ValueError("Provide an image or a dataset index.")
    p = softmax(network.logits(x) / metadata["temperature"])[0]
    return {"digit": int(p.argmax()), "confidence": float(p.max()),
            "probabilities": p.tolist()}
