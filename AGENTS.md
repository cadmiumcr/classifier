# CADMIUM CLASSIFIER

**ML classifiers** - Bayes, Viterbi (HMM), tabular (KNN, LogReg).

## OVERVIEW

Three classifier types: text classification (Bayes), sequence labeling (Viterbi/HMM), numerical features (KNN, LogisticRegression). Uses `num.cr` for matrix ops and `msgpack` for model persistence.

## STRUCTURE

```
src/cadmium/classifier/
├── classifier.cr           # Base class
├── bayes.cr               # Naive Bayes (Laplace smoothing)
├── viterbi.cr             # HMM sequence labeling
├── logistic_regression.cr # Binary classification
└── tabular/
    ├── knn.cr             # K-Nearest Neighbors
    ├── logistic_regression.cr
    └── distance_metrics.cr # Euclidean, Manhattan, Chebyshev, Cosine
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Text classification | `bayes.cr` | Bag-of-words, train/classify |
| POS tagging / sequences | `viterbi.cr` | Hidden Markov Model |
| Numerical features | `tabular/knn.cr` | Feature vectors, distance metrics |
| Model save/load | Any classifier | `.to_msgpack` / `.from_msgpack` |

## KEY CLASSES

| Class | Input | Output |
|-------|-------|--------|
| `Bayes` | text string | category probabilities |
| `Viterbi` | token sequence | tag sequence |
| `Tabular::KNN` | Float64 feature vector | category string |
| `Tabular::LogisticRegression` | Float64 feature vector | category + probabilities |

## CONVENTIONS

- All classifiers: `train(...)` then `classify(...)`
- Bayes uses `Cadmium::Tokenizer::Word` by default (configurable)
- Tabular classifiers: numerical features only (one-hot encode categoricals)
- Model serialization: prefer msgpack (3-5x smaller than JSON)

## EXTERNAL DEPS

| Dep | Purpose |
|-----|---------|
| `num.cr` | Matrix operations for Viterbi algorithm |
| `msgpack` | Binary model serialization |
| `cadmium_tokenizer` | Text tokenization for Bayes |

## ANTI-PATTERNS

| Pattern | Why Forbidden |
|---------|---------------|
| JSON for large models | Use msgpack (5-10x faster, 3-5x smaller) |
| Raw feature vectors without scaling | Normalize before training (esp. KNN) |
| Categorical features in tabular | One-hot encode first |

## USAGE PATTERNS

```crystal
# Text classification
bayes = Bayes.new
bayes.train("great product!", "positive")
bayes.classify("awesome!")  # => {"positive" => 95.0, ...}

# Sequence labeling
viterbi = Viterbi.new
viterbi.train([{"word", "tag"}, ...])
viterbi.classify(["word1", "word2"])  # => {"word1" => "tag", ...}

# Numerical classification
knn = Tabular::KNN.new(k: 3)
knn.train([[1.0, 2.0], [5.0, 6.0]], ["a", "b"])
knn.classify([1.5, 2.5])  # => "a"
```

## MODEL PERSISTENCE

```crystal
# Save (prefer msgpack)
File.write("model.bin", classifier.to_msgpack)

# Load
classifier = Bayes.from_msgpack(File.read("model.bin"))
```
