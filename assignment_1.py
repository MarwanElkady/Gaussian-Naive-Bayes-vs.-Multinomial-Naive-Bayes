# IMPORTS

import numpy as np
import pandas as pd
import math
import os
import re
import string
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_mean(values):
    """
    μ = (1/n) * Σ xi
    Accepts a list or 1-D numpy array.  Returns a Python float.
    """
    n = len(values)
    if n == 0:
        raise ValueError("compute_mean: empty sequence")
    total = 0.0
    for x in values:
        total += x
    return total / n


def compute_variance(values):
    """
    σ² = (1/n) * Σ (xi − μ)²
    Population variance (matches the Naive Bayes literature).
    """
    n = len(values)
    if n == 0:
        raise ValueError("compute_variance: empty sequence")
    mu = compute_mean(values)
    total = 0.0
    for x in values:
        total += (x - mu) ** 2
    return total / n


def compute_accuracy(y_true, y_pred):
    """
    Accuracy = correct_predictions / total_samples
    Both arguments must be same-length iterables.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("compute_accuracy: length mismatch")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — GAUSSIAN NAIVE BAYES (Abalone Dataset)
# ═════════════════════════════════════════════════════════════════════════════

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes from scratch.

    Supports:
        fit(X, y)                   — train the model
        predict(x)                  — class label for a single sample
        predict_prob(x)             — dict {class: posterior prob} for a sample
        predict_batch(X)            — class labels for a 2-D array
        predict_batch_log(X)        — same but using log-probabilities
    """

    def __init__(self, use_log: bool = False):
        self.use_log = use_log
        self.classes_      = None   # unique class labels
        self.priors_       = {}     # P(C)
        self.means_        = {}     # {class: [μ1, μ2, …]}
        self.variances_    = {}     # {class: [σ²1, σ²2, …]}
        self.log_priors_   = {}     # log P(C)

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        X : 2-D array-like  (n_samples × n_features)
        y : 1-D array-like  (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples = len(y)
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            X_cls = X[y == cls]
            n_cls = len(X_cls)

            # Prior
            self.priors_[cls]     = n_cls / n_samples
            self.log_priors_[cls] = math.log(self.priors_[cls])

            # Per-feature statistics (using OUR custom functions)
            self.means_[cls]     = [compute_mean(X_cls[:, j])     for j in range(X.shape[1])]
            self.variances_[cls] = [compute_variance(X_cls[:, j]) for j in range(X.shape[1])]

        return self

    # ── Gaussian PDF ──────────────────────────────────────────────────────────
    @staticmethod
    def _gaussian_pdf(x, mean, var):
        """P(xi | C) under Gaussian assumption."""
        if var == 0.0:
            # avoid division-by-zero; treat as a near-zero spike
            var = 1e-9
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    @staticmethod
    def _log_gaussian_pdf(x, mean, var):
        """log P(xi | C)"""
        if var == 0.0:
            var = 1e-9
        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    # ── Raw (non-log) posterior ───────────────────────────────────────────────
    def _raw_posteriors(self, x):
        """Returns {class: unnormalised posterior} using plain multiplication."""
        posteriors = {}
        for cls in self.classes_:
            p = self.priors_[cls]
            for j, xj in enumerate(x):
                p *= self._gaussian_pdf(xj, self.means_[cls][j], self.variances_[cls][j])
            posteriors[cls] = p
        return posteriors

    # ── Log posterior ─────────────────────────────────────────────────────────
    def _log_posteriors(self, x):
        """Returns {class: log posterior}"""
        log_posts = {}
        for cls in self.classes_:
            lp = self.log_priors_[cls]
            for j, xj in enumerate(x):
                lp += self._log_gaussian_pdf(xj, self.means_[cls][j], self.variances_[cls][j])
            log_posts[cls] = lp
        return log_posts

    # ── Public API — single sample ────────────────────────────────────────────
    def predict(self, x):
        """
        Predict the class label for ONE sample x.
        Uses log-probabilities when self.use_log is True.
        """
        x = list(x)
        if self.use_log:
            scores = self._log_posteriors(x)
        else:
            scores = self._raw_posteriors(x)
        return max(scores, key=scores.get)

    def predict_prob(self, x):
        """
        Return a dict {class_label: probability} for ONE sample x.
        Probabilities are normalised so they sum to 1.
        Always uses raw (non-log) probabilities for interpretability.
        """
        x = list(x)
        raw = self._raw_posteriors(x)
        total = sum(raw.values())
        if total == 0:
            n = len(self.classes_)
            return {cls: 1.0 / n for cls in self.classes_}
        return {cls: v / total for cls, v in raw.items()}

    # ── Public API — batch ────────────────────────────────────────────────────
    def predict_batch(self, X):
        """Predict class labels for a 2-D array (plain probabilities)."""
        X = np.array(X, dtype=float)
        orig_log = self.use_log
        self.use_log = False
        preds = [self.predict(X[i]) for i in range(len(X))]
        self.use_log = orig_log
        return np.array(preds)

    def predict_batch_log(self, X):
        """Predict class labels for a 2-D array (log probabilities)."""
        X = np.array(X, dtype=float)
        orig_log = self.use_log
        self.use_log = True
        preds = [self.predict(X[i]) for i in range(len(X))]
        self.use_log = orig_log
        return np.array(preds)


# ── Abalone helpers ───────────────────────────────────────────────────────────

def load_abalone(csv_path: str):
    """
    Load the Abalone CSV, drop the Sex column, map Rings → age class.
    Returns (X_df, y_series).
    """
    df = pd.read_csv(csv_path)

    # ── Normalise column names (Kaggle file uses Title Case) ─────────────────
    df.columns = [c.strip() for c in df.columns]

    # Drop categorical sex column
    if "Sex" in df.columns:
        df = df.drop(columns=["Sex"])

    # Map rings to age classes
    def rings_to_class(r):
        if r <= 8:
            return "Young"
        elif r <= 11:
            return "Adult"
        else:
            return "Old"

    feature_cols = [c for c in df.columns if c.lower() != "rings"]
    df["AgeClass"] = df["Rings"].apply(rings_to_class)

    X = df[feature_cols].values.astype(float)
    y = df["AgeClass"].values
    return X, y, feature_cols


def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """80 / 20 split — no sklearn."""
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_gaussian_nb(csv_path: str):
    print("\n" + "═" * 70)
    print("  PART 2 — GAUSSIAN NAIVE BAYES  (Abalone Dataset)")
    print("═" * 70)

    X, y, feature_names = load_abalone(csv_path)
    print(f"  Dataset shape : {X.shape}  |  Classes : {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    print(f"  Train samples : {len(y_train)}  |  Test samples : {len(y_test)}")

    # ── Train (plain) ─────────────────────────────────────────────────────────
    gnb = GaussianNaiveBayes(use_log=False)
    gnb.fit(X_train, y_train)

    y_pred_plain = gnb.predict_batch(X_test)
    acc_plain    = compute_accuracy(y_test, y_pred_plain)

    # ── Train (log) ───────────────────────────────────────────────────────────
    y_pred_log = gnb.predict_batch_log(X_test)
    acc_log    = compute_accuracy(y_test, y_pred_log)

    print(f"\n  Accuracy (plain probabilities) : {acc_plain:.4f} ({acc_plain*100:.2f}%)")
    print(f"  Accuracy (log   probabilities) : {acc_log:.4f}  ({acc_log*100:.2f}%)")

    # ── Demo: predict a single sample ─────────────────────────────────────────
    sample = X_test[0]
    print(f"\n  Single sample prediction → predict()      : {gnb.predict(sample)}")
    probs = gnb.predict_prob(sample)
    print(f"  Single sample prediction → predict_prob() :")
    for cls, prob in sorted(probs.items()):
        print(f"      {cls:<8} : {prob:.4f}")

    # ── Part 5 visualisation A — feature distributions ────────────────────────
    plot_feature_distributions(X_train, y_train, feature_names)

    return gnb, X_train, y_train, feature_names


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — MULTINOMIAL NAIVE BAYES (IMDB Reviews)
# ═════════════════════════════════════════════════════════════════════════════

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes for text classification, built from scratch.

    Steps:
        1. preprocess(text)              — lowercase, remove punctuation, tokenise
        2. build_vocabulary(corpus)      — vocabulary dict from training tokens
        3. bag_of_words(tokens)          — count vector over the vocabulary
        4. fit(texts, labels)            — compute priors + word probs (Laplace)
        5. predict(text)                 — class label
        6. predict_prob(text)            — normalised posterior per class
        7. predict_batch / _log variants — batch helpers
    """

    def __init__(self, use_log: bool = True):
        self.use_log   = use_log
        self.vocab_    = {}          # word → index
        self.classes_  = None
        self.priors_   = {}
        self.log_priors_ = {}
        # word_probs_[cls][word_idx] = P(word | cls)  with Laplace smoothing
        self.word_probs_    = {}
        self.log_word_probs_ = {}

    # ── Text preprocessing ────────────────────────────────────────────────────
    @staticmethod
    def preprocess(text: str):
        """Lowercase → remove punctuation → tokenise on whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        return tokens

    # ── Vocabulary ────────────────────────────────────────────────────────────
    def build_vocabulary(self, corpus):
        """
        corpus : iterable of raw text strings (training set only).
        Builds self.vocab_  {word: index}
        """
        word_set = set()
        for text in corpus:
            for tok in self.preprocess(text):
                word_set.add(tok)
        self.vocab_ = {w: i for i, w in enumerate(sorted(word_set))}
        return self.vocab_

    # ── Bag-of-words ─────────────────────────────────────────────────────────
    def bag_of_words(self, tokens):
        """
        tokens : list of preprocessed tokens for ONE document.
        Returns a 1-D numpy array of length |vocab|.
        Only counts words that appear in the vocabulary.
        """
        bow = np.zeros(len(self.vocab_), dtype=np.int32)
        for tok in tokens:
            if tok in self.vocab_:
                bow[self.vocab_[tok]] += 1
        return bow

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, texts, labels):
        """
        texts  : list of raw review strings
        labels : list/array of class labels (e.g. 'pos', 'neg')
        """
        labels = np.array(labels)
        self.classes_ = np.unique(labels)
        n_samples = len(labels)
        V = len(self.vocab_)

        for cls in self.classes_:
            mask = labels == cls
            n_cls = mask.sum()

            # Prior
            self.priors_[cls]      = n_cls / n_samples
            self.log_priors_[cls]  = math.log(self.priors_[cls])

            # Aggregate word counts for this class
            word_counts = np.zeros(V, dtype=np.float64)
            for text in np.array(texts)[mask]:
                tokens = self.preprocess(text)
                word_counts += self.bag_of_words(tokens)

            # Laplace smoothing: P(w|C) = (count(w,C) + 1) / (Σ counts + V)
            total = word_counts.sum() + V
            self.word_probs_[cls]     = (word_counts + 1.0) / total
            self.log_word_probs_[cls] = np.log(self.word_probs_[cls])

        return self

    # ── Scoring helpers ───────────────────────────────────────────────────────
    def _score_plain(self, bow):
        scores = {}
        for cls in self.classes_:
            p = self.priors_[cls]
            for idx, count in enumerate(bow):
                if count > 0:
                    p *= self.word_probs_[cls][idx] ** count
            scores[cls] = p
        return scores

    def _score_log(self, bow):
        scores = {}
        for cls in self.classes_:
            lp = self.log_priors_[cls]
            lp += float(np.dot(bow, self.log_word_probs_[cls]))
            scores[cls] = lp
        return scores

    # ── Public API — single sample ────────────────────────────────────────────
    def predict(self, text: str):
        tokens = self.preprocess(text)
        bow    = self.bag_of_words(tokens)
        if self.use_log:
            scores = self._score_log(bow)
        else:
            scores = self._score_plain(bow)
        return max(scores, key=scores.get)

    def predict_prob(self, text: str):
        """Returns normalised posterior dict."""
        tokens = self.preprocess(text)
        bow    = self.bag_of_words(tokens)
        raw    = self._score_plain(bow)
        total  = sum(raw.values())
        if total == 0:
            n = len(self.classes_)
            return {cls: 1.0 / n for cls in self.classes_}
        return {cls: v / total for cls, v in raw.items()}

    # ── Batch ─────────────────────────────────────────────────────────────────
    def predict_batch(self, texts):
        return np.array([self.predict(t) for t in texts])

    # ── Word-frequency helpers (for visualisation) ────────────────────────────
    def top_words(self, cls, n=20):
        """Return the n most probable words for a class (excluding smoothing artefacts)."""
        probs = self.word_word_probs_per_class(cls)
        sorted_words = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_words[:n]

    def word_word_probs_per_class(self, cls):
        idx_to_word = {v: k for k, v in self.vocab_.items()}
        return {idx_to_word[i]: self.word_probs_[cls][i] for i in range(len(self.vocab_))}


# ── IMDB loading helpers ──────────────────────────────────────────────────────

def load_imdb_from_folder(base_dir: str, split: str, max_per_class: int = None):
    """
    Loads IMDB reviews from the aclImdb directory structure.
    base_dir : path to the extracted aclImdb folder
    split    : 'train' or 'test'
    Returns  : (texts, labels)  — lists of equal length
    """
    texts, labels = [], []
    for sentiment in ("pos", "neg"):
        folder = os.path.join(base_dir, split, sentiment)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder: {folder}")
        files = sorted(os.listdir(folder))
        if max_per_class is not None:
            files = files[:max_per_class]
        for fname in files:
            if fname.endswith(".txt"):
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as fh:
                    texts.append(fh.read())
                labels.append(sentiment)
    return texts, labels


def run_multinomial_nb(imdb_dir: str, max_per_class: int = None):
    print("\n" + "═" * 70)
    print("  PART 3 — MULTINOMIAL NAIVE BAYES  (IMDB Dataset)")
    print("═" * 70)

    print("  Loading training data…")
    train_texts, train_labels = load_imdb_from_folder(imdb_dir, "train", max_per_class)
    print("  Loading test data…")
    test_texts,  test_labels  = load_imdb_from_folder(imdb_dir, "test",  max_per_class)

    print(f"  Train : {len(train_texts)} reviews  |  Test : {len(test_texts)} reviews")

    mnb = MultinomialNaiveBayes(use_log=True)

    print("  Building vocabulary…")
    mnb.build_vocabulary(train_texts)
    print(f"  Vocabulary size : {len(mnb.vocab_):,}")

    print("  Fitting model…")
    mnb.fit(train_texts, train_labels)

    # ── Accuracy with log probs ───────────────────────────────────────────────
    print("  Predicting test set (log probs)…")
    mnb.use_log = True
    y_pred_log = mnb.predict_batch(test_texts)
    acc_log    = compute_accuracy(test_labels, y_pred_log)

    # ── Accuracy with plain probs ─────────────────────────────────────────────
    print("  Predicting test set (plain probs)…")
    mnb.use_log = False
    y_pred_plain = mnb.predict_batch(test_texts)
    acc_plain    = compute_accuracy(test_labels, y_pred_plain)

    print(f"\n  Accuracy (log   probabilities) : {acc_log:.4f}  ({acc_log*100:.2f}%)")
    print(f"  Accuracy (plain probabilities) : {acc_plain:.4f} ({acc_plain*100:.2f}%)")

    # ── Part 5 visualisation B — top words ───────────────────────────────────
    plot_top_words(mnb)

    return mnb


# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Young"  : "#4ECDC4",
    "Adult"  : "#FFE66D",
    "Old"    : "#FF6B6B",
    "pos"    : "#06D6A0",
    "neg"    : "#EF476F",
    "bg"     : "#0F1117",
    "panel"  : "#1A1D27",
    "text"   : "#E8EAF0",
    "grid"   : "#2A2D3A",
}


def plot_feature_distributions(X_train, y_train, feature_names):
    """
    Histogram of each feature, coloured by age class.
    Saves → gaussian_feature_distributions.png
    """
    classes = np.unique(y_train)
    n_feat  = len(feature_names)
    cols    = 4
    rows    = math.ceil(n_feat / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for j, ax in enumerate(axes_flat):
        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

        if j < n_feat:
            for cls in classes:
                vals = X_train[y_train == cls, j]
                ax.hist(vals, bins=30, alpha=0.65, color=PALETTE[cls],
                        edgecolor="none", label=cls)
            ax.set_title(feature_names[j], color=PALETTE["text"], fontsize=10, pad=6)
            ax.tick_params(colors=PALETTE["grid"], labelcolor=PALETTE["text"], labelsize=7)
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=0.5)
        else:
            ax.set_visible(False)

    # shared legend
    handles = [mpatches.Patch(color=PALETTE[c], label=c) for c in classes]
    fig.legend(handles=handles, loc="lower right", framealpha=0.3,
               fontsize=10, facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])

    fig.suptitle("Abalone — Feature Distributions by Age Class",
                 color=PALETTE["text"], fontsize=14, y=1.01)
    plt.tight_layout()

    out = "gaussian_feature_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"\n  [✓] Saved → {out}")


def plot_top_words(mnb: MultinomialNaiveBayes, n=20):
    """
    Horizontal bar chart — top-20 words per sentiment class.
    Saves → mnb_top_words.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, cls, color in zip(axes, ["pos", "neg"], [PALETTE["pos"], PALETTE["neg"]]):
        top = mnb.top_words(cls, n=n)
        words  = [w for w, _ in reversed(top)]
        probs  = [p for _, p in reversed(top)]

        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

        bars = ax.barh(words, probs, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(f"Top {n} Words — {'Positive' if cls=='pos' else 'Negative'}",
                     color=PALETTE["text"], fontsize=12, pad=10)
        ax.tick_params(colors=PALETTE["grid"], labelcolor=PALETTE["text"], labelsize=9)
        ax.set_xlabel("P(word | class)", color=PALETTE["text"], fontsize=9)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5, alpha=0.5)

    fig.suptitle("IMDB — Top 20 Words per Sentiment Class",
                 color=PALETTE["text"], fontsize=14, y=1.01)
    plt.tight_layout()

    out = "mnb_top_words.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [✓] Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── PATHS — edit these two lines to point at your downloaded datasets ──────
    ABALONE_CSV = r"M:\College\THIRD YEAR\second semmster\COGNITIVE\Assignment 1\Abalone\Abalone.csv"          # Kaggle Abalone CSV
    IMDB_DIR    = "aclImdb"              # extracted aclImdb folder

    # Optional: cap IMDB at N reviews per class to speed up testing
    # Set to None to use all 25 000 reviews per split.
    IMDB_MAX_PER_CLASS = None            # e.g. 2000 for a quick test

    # ── Gaussian NB ───────────────────────────────────────────────────────────
    if not os.path.isfile(ABALONE_CSV):
        print(f"[!] Abalone CSV not found at '{ABALONE_CSV}'. "
              "Download from https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset")
    else:
        run_gaussian_nb(ABALONE_CSV)

    # ── Multinomial NB ────────────────────────────────────────────────────────
    if not os.path.isdir(IMDB_DIR):
        print(f"\n[!] IMDB folder not found at '{IMDB_DIR}'. "
              "Download & extract https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
    else:
        run_multinomial_nb(IMDB_DIR, max_per_class=IMDB_MAX_PER_CLASS)