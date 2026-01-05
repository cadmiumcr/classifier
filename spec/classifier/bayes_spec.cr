require "../spec_helper"

describe Cadmium::Classifier::Bayes do
  subject = Cadmium::Classifier::Bayes

  describe "#initialize" do
    it "successfully initalizes all defaults" do
      classifier = subject.new
      classifier.tokenizer.should be_a(Cadmium::Tokenizer::Word)
      classifier.vocabulary.should eq(Set(String).new)
      classifier.vocabulary_size.should eq(0)
      classifier.total_documents.should eq(0)
      classifier.doc_count.should eq({} of String => Int32)
      classifier.word_count.should eq({} of String => Int32)
      classifier.word_frequency_count.should eq({} of String => Hash(String, Int32))
      classifier.categories.should eq([] of String)
    end

    it "uses a custom tokenizer" do
      classifier = subject.new(tokenizer: Cadmium::Tokenizer::Aggressive.new(lang: :en))
      classifier.tokenizer.should be_a(Cadmium::Tokenizer::Aggressive)
    end
  end

  describe "#train" do
    it "adds categories" do
      classifier = subject.new
      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")
      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")
      classifier.categories.should contain("programming")
      classifier.categories.should contain("off-topic")
    end

    it "increases the total_documents count" do
      classifier = subject.new
      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")

      classifier.total_documents.should eq(2)

      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")

      classifier.total_documents.should eq(4)
    end

    it "adds words to the vocabulary set" do
      classifier = subject.new
      classifier.train("crystal is an awesome programming language", "programming")
      expected_vocab = Set{"crystal", "is", "an", "awesome", "programming", "language"}
      classifier.vocabulary.should eq(expected_vocab)
    end
  end

  describe "#classify" do
    it "correctly classifys `positive` and `negative` categories" do
      classifier = subject.new

      # teach it positive phrases
      classifier.train("amazing, awesome movie!! Yeah!!", "positive")
      classifier.train("Sweet, this is incredibly, amazing, perfect, great!!", "positive")

      # teach it a negative phrase
      classifier.train("terrible, shitty thing. Damn. Sucks!!", "negative")

      # teach it a neutral phrase
      classifier.train("I dont really know what to make of this.", "neutral")

      classifier.classify_category("awesome, cool, amazing!! Yay.").should eq("positive")
      classifier.classify_category("This is a damn shitty awful thing!").should eq("negative")
    end

    it "correctly classifys `programming` and `off-topic`" do
      classifier = subject.new

      # some programming data
      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")

      # some off topic data
      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")

      classifier.classify_category("this post is about crystal").should eq("programming")
      classifier.classify_category("i don't know what I'm about").should eq("off-topic")
    end

    it "handles unicode characters" do
      classifier = subject.new

      classifier.train("Omg I love you so much 💕", "positive")
      classifier.train("You're the best! 😍😄", "positive")
      classifier.train("Damn you suck 👎", "negative")
      classifier.train("You are such a 💩 head", "negative")

      classifier.classify_category("I love you 💕").should eq("positive")
      classifier.classify_category("This sucks 👎").should eq("negative")
    end

    it "returns all probabilities with highest first" do
      classifier = subject.new

      classifier.train("I love this!", "positive")
      classifier.train("This is great", "positive")
      classifier.train("I hate this", "negative")

      result = classifier.classify("This is amazing!")

      result.should be_a(Hash(String, Float64))
      result.keys.should contain("positive")
      result.keys.should contain("negative")

      # Highest probability should be "positive"
      result.first_key.should eq("positive")
      result["positive"].should be > result["negative"]
    end
  end

  describe "json serialization and deserialization" do
    it "exports and imports a trained set correctly" do
      classifier = subject.new

      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")
      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")

      json = classifier.to_json
      restored = subject.from_json(json)

      restored.total_documents.should eq(4)
      restored.vocabulary_size.should eq(25)
      restored.categories.should contain("programming")
      restored.categories.should contain("off-topic")
    end
  end

  describe "yaml serialization and deserialization" do
    it "exports and imports a trained set correctly" do
      classifier = subject.new

      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")
      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")

      yaml = classifier.to_yaml
      restored = subject.from_yaml(yaml)

      restored.total_documents.should eq(4)
      restored.vocabulary_size.should eq(25)
      restored.categories.should contain("programming")
      restored.categories.should contain("off-topic")
    end
  end

  describe "messagepack serialization and deserialization" do
    it "exports and imports a trained set correctly" do
      classifier = subject.new

      classifier.train("crystal is an awesome programming language", "programming")
      classifier.train("ruby is nice, but not as fast as crystal", "programming")
      classifier.train("my wife and I went to the beach", "off-topic")
      classifier.train("my dog likes to go outside and play", "off-topic")

      # Test serialization
      bytes = classifier.to_msgpack
      bytes.should be_a(Bytes)

      # Test deserialization
      restored = subject.from_msgpack(bytes)

      restored.total_documents.should eq(4)
      restored.vocabulary_size.should eq(25)
      restored.categories.should contain("programming")
      restored.categories.should contain("off-topic")
    end

    it "preserves classification accuracy after serialization roundtrip" do
      classifier = subject.new

      classifier.train("I love this product!", "positive")
      classifier.train("This is amazing", "positive")
      classifier.train("I hate this", "negative")
      classifier.train("This is terrible", "negative")

      # Get original classification
      original_result = classifier.classify("This is great!")

      # Serialize and deserialize
      bytes = classifier.to_msgpack
      restored = subject.from_msgpack(bytes)

      # Classification should be identical
      restored_result = restored.classify("This is great!")

      restored_result["positive"].should be_close(original_result["positive"], 0.001)
      restored_result["negative"].should be_close(original_result["negative"], 0.001)
    end

    it "produces more compact output than JSON" do
      classifier = subject.new

      # Train with some data
      100.times do
        classifier.train("This is a sample text with some words", "category")
      end

      json_bytes = classifier.to_json.bytesize
      msgpack_bytes = classifier.to_msgpack.bytesize

      # MessagePack should be more compact
      msgpack_bytes.should be < json_bytes
    end

    it "excludes tokenizer from serialization" do
      custom_tokenizer = Cadmium::Tokenizer::Aggressive.new(lang: :en)
      classifier = subject.new(tokenizer: custom_tokenizer)

      classifier.train("test data here", "test")

      bytes = classifier.to_msgpack
      restored = subject.from_msgpack(bytes)

      # Restored classifier should use default tokenizer
      restored.tokenizer.should be_a(Cadmium::Tokenizer::Word)
    end
  end
end
