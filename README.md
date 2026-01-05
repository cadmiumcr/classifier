# Classifier

Cadmium comes with two classifiers so far, a Classic Bayes classifier and a Viterbi classifier.

Those are probabalistic classifiers that, when trained with a data set, can classify words (or other tokens) according to categories.

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

## Contributing

1. Fork it (<https://github.com/cadmiumcr/classifier/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Watson](https://github.com/watzon) - creator and maintainer
