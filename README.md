# Classifier

Cadmium comes with classifiers for different types of data:

- **Bayes**: Text classification using bag-of-words
- **Viterbi**: Sequence labeling using Hidden Markov Models
- **Tabular**: Multi-feature numerical classification (K-Nearest Neighbors, Logistic Regression)

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     cadmium_classifier:
       github: cadmiumcr/classifier
   ```

2. Run `shards install`

## Usage

### Bayes Classifier

The Bayes classifier returns a hash with all categories and their probabilities (sorted from highest to lowest):

```crystal
require "cadmium_classifier"

classifier = Cadmium::Classifier::Bayes.new

classifier.train("crystal is an awesome programming language", "programming")
classifier.train("ruby is nice, but not as fast as crystal", "programming")

classifier.train("my wife and I went to the beach", "off-topic")
classifier.train("my dog likes to go outside and play", "off-topic")

classifier.classify("Crystal is my favorite!")
# => {"programming" => 91.06, "off-topic" => 8.94}
```

If you only need the top category, use `classify_category`:

```crystal
classifier.classify_category("Crystal is my favorite!")
# => "programming"
```

#### Saving and Loading

**Recommended: MessagePack (binary format)**

MessagePack is the most efficient format - 3-5x smaller and 5-10x faster than JSON/YAML:

```crystal
# Export to binary format
bytes = classifier.to_msgpack
File.write("classifier.model", bytes)

# Import from binary format
bytes = File.read("classifier.model")
classifier = Cadmium::Classifier::Bayes.from_msgpack(bytes)
```

**JSON format:**

```crystal
require "json"
json = classifier.to_json
File.write("classifier.json", json)

# Later
json = File.read("classifier.json")
classifier = Cadmium::Classifier::Bayes.from_json(json)
```

**YAML format:**

```crystal
require "yaml"
yaml = classifier.to_yaml
File.write("classifier.yaml", yaml)

# Later
classifier = Cadmium::Classifier::Bayes.from_yaml(File.read("classifier.yaml"))
```

### Viterbi Classifier

The Viterbi classifier is a Hidden Markov Model classifier:

```crystal
require "cadmium_classifier"

classifier = Cadmium::Classifier::Viterbi.new

training_data = [
  {"they", "pronoun"},
  {"drink", "verb"},
  {"water", "verb"},
]

classifier.train(training_data)
result = classifier.classify(["they", "drink", "water"])
# => {"they" => "pronoun", "drink" => "verb", "water" => "verb"}
```

### Tabular Classifiers

Tabular classifiers work with numerical feature vectors for multi-dimensional classification tasks.

#### K-Nearest Neighbors (KNN)

KNN stores all training data and classifies by finding the k nearest neighbors:

```crystal
require "cadmium_classifier"

classifier = Cadmium::Classifier::Tabular::KNN.new(k: 3)

# Training data with 3 features per sample
features = [
  [1.0, 2.0, 3.0],
  [1.1, 2.1, 3.1],
  [5.0, 6.0, 7.0],
]
labels = ["class_a", "class_a", "class_b"]

classifier.train(features, labels)

# Predict new sample
result = classifier.classify([1.05, 2.05, 3.05])
# => "class_a"

# Get detailed results with vote counts
details = classifier.classify_details([1.05, 2.05, 3.05])
# => {"class_a" => 3, "class_b" => 0}
```

**Distance Metrics:**

KNN supports multiple distance metrics:

```crystal
# Euclidean (default)
knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)

# Manhattan
knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)

# Chebyshev
knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Chebyshev)

# Cosine
knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
```

#### Logistic Regression

Logistic Regression uses gradient descent for binary classification with probabilistic output:

```crystal
require "cadmium_classifier"

classifier = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.01, max_iterations: 1000)

features = [
  [1.0, 2.0, 3.0],
  [1.1, 2.1, 3.1],
  [5.0, 6.0, 7.0],
]
labels = ["class_a", "class_a", "class_b"]

classifier.train(features, labels)

# Predict new sample
result = classifier.classify([1.05, 2.05, 3.05])
# => "class_a"

# Get probability scores
probs = classifier.classify_probabilities([1.05, 2.05, 3.05])
# => {"class_a" => 85.5, "class_b" => 14.5}
```

**Saving and Loading Tabular Models:**

```crystal
# Save
classifier.save_model("model.msgpack")

# Load
loaded = Cadmium::Classifier::Tabular::KNN.load_model("model.msgpack")
# or
loaded = Cadmium::Classifier::Tabular::LogisticRegression.load_model("model.msgpack")
```

#### Example: Fraud Detection

```crystal
# Features: [amount, merchant_distance, time_diff, ...]
features = [
  [10.50, 0.5, 1.2, 0.0, 0.0, 1.0, 0.8, 0.5, 0.1],   # legit
  [25.00, 1.2, 0.5, 0.0, 0.0, 1.0, 0.9, 0.6, 0.2],   # legit
  [5000.00, 150.0, 100.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.95], # fraud
  [2500.00, 200.0, 50.0, 1.0, 1.0, 0.0, 0.15, 0.1, 0.9],  # fraud
]
labels = ["legit", "legit", "fraud", "fraud"]

classifier = Cadmium::Classifier::Tabular::KNN.new(k: 1)
classifier.train(features, labels)

# Classify new transaction
new_transaction = [5000.00, 180.0, 120.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.92]
classifier.classify(new_transaction)
# => "fraud"
```

**Important Notes:**

- **Numerical features only**: Categorical features require one-hot encoding
- **Feature scaling**: Normalize/scale features before training for better results
- **KNN**: O(n) prediction time, suitable for small-to-medium datasets
- **Logistic Regression**: O(1) prediction time, better for large datasets

## Contributing

1. Fork it (<https://github.com/cadmiumcr/classifier/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Watson](https://github.com/watzon) - creator and maintainer
