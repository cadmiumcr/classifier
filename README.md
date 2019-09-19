# Classifier

Cadmium comes with one classifier so far, a Classic Bayes classifier. It is a probabalistic classifier that, when trained with a data set, can classify words according to categories.

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     cadmium_classifier:
       github: cadmiumcr/classifier
   ```

2. Run `shards install`

## Usage

```crystal
require "cadmium_classifier"
```

```crystal
classifier = Cadmium::Classifier::Bayes.new

classifier.train("crystal is an awesome programming language", "programming")
classifier.train("ruby is nice, but not as fast as crystal", "programming")

classifier.train("my wife and I went to the beach", "off-topic")
classifier.train("my dog likes to go outside and play", "off-topic")

classifier.classify("Crystal is my favorite!")
# => "programming"
```

You can save the classifier as JSON as well

```crystal
require "json"
json = classifier.to_json
File.write("classifier.json", json)
```

And load it again later

```crystal
require "json"
json = File.open("classifier.json")
classifier = classifier.from_json(json)
```


## Contributing

1. Fork it (<https://github.com/cadmiumcr/lassifier/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Watson](https://github.com/watzon) - creator and maintainer
