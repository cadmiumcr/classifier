require "../../spec_helper"

describe Cadmium::Classifier::Tabular::LogisticRegression do
  describe "#initialize" do
    it "initializes with default values" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      lr.learning_rate.should eq(0.01)
      lr.max_iterations.should eq(1000)
    end

    it "initializes with custom values" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 500)
      lr.learning_rate.should eq(0.1)
      lr.max_iterations.should eq(500)
    end
  end

  describe "#train" do
    it "trains on simple binary classification data" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 1000)

      # Linearly separable data
      features = [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [6.0, 6.0],
        [5.1, 5.1],
      ]
      labels = ["a", "a", "a", "b", "b", "b"]

      lr.train(features, labels)

      # Should classify correctly
      lr.classify([0.0, 0.0]).should eq("a")
      lr.classify([5.0, 5.0]).should eq("b")
    end

    it "raises on empty data" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      expect_raises(ArgumentError, "Cannot train on empty data") do
        lr.train([] of Array(Float64), [] of String)
      end
    end

    it "raises on non-binary classification" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      features = [[1.0], [2.0], [3.0]]
      labels = ["a", "b", "c"] # 3 classes

      expect_raises(ArgumentError, "Logistic regression requires exactly 2 classes") do
        lr.train(features, labels)
      end
    end

    it "raises on mismatched feature and label counts" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      features = [[1.0], [2.0]]
      labels = ["a"]

      expect_raises(ArgumentError, "Features and labels must have same length") do
        lr.train(features, labels)
      end
    end
  end

  describe "#classify" do
    it "classifies linearly separable data correctly" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 1000)

      # Clear separation
      features = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
      ]
      labels = ["a", "a", "a", "b", "b", "b"]

      lr.train(features, labels)

      # Points in "a" region
      lr.classify([0.5, 0.5]).should eq("a")
      lr.classify([0.0, 0.5]).should eq("a")

      # Points in "b" region
      lr.classify([10.5, 10.5]).should eq("b")
      lr.classify([11.0, 10.0]).should eq("b")
    end

    it "raises if not trained" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      expect_raises(ArgumentError, "Classifier has not been trained") do
        lr.classify([1.0, 2.0])
      end
    end

    it "raises on feature dimension mismatch" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 100)
      lr.train([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])

      expect_raises(ArgumentError, "Feature dimension mismatch") do
        lr.classify([1.0, 2.0, 3.0])
      end
    end
  end

  describe "#classify_probabilities" do
    it "returns probability scores for both classes" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 1000)

      features = [
        [0.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [10.0, 11.0],
      ]
      labels = ["a", "a", "b", "b"]

      lr.train(features, labels)

      # Point clearly in "a" region
      probs = lr.classify_probabilities([0.0, 0.0])
      probs["a"].should be > 50.0 # More than 50% confidence for "a"
      probs["b"].should be < 50.0

      # Point clearly in "b" region
      probs = lr.classify_probabilities([10.0, 10.0])
      probs["b"].should be > 50.0 # More than 50% confidence for "b"
      probs["a"].should be < 50.0
    end

    it "probabilities sum to 100" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 500)

      features = [
        [0.0, 0.0],
        [10.0, 10.0],
      ]
      labels = ["a", "b"]

      lr.train(features, labels)

      probs = lr.classify_probabilities([5.0, 5.0]) # Somewhere in the middle
      (probs["a"] + probs["b"]).should be_close(100.0, 0.01)
    end
  end

  describe "#classify_batch" do
    it "classifies multiple samples" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 1000)

      features = [
        [0.0, 0.0],
        [10.0, 10.0],
      ]
      labels = ["a", "b"]

      lr.train(features, labels)

      results = lr.classify_batch([[0.5, 0.5], [10.5, 10.5]])
      results.should eq(["a", "b"])
    end
  end

  describe "#weights and #bias" do
    it "returns the learned weights and bias" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 100)

      features = [[0.0], [1.0], [2.0]]
      labels = ["a", "a", "b"]

      lr.train(features, labels)

      weights = lr.weights
      weights.size.should eq(1) # 1 feature

      bias = lr.bias
      bias.should be_a(Float64)
    end

    it "raises if not trained" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new
      expect_raises(ArgumentError, "Classifier has not been trained") do
        lr.weights
      end
    end
  end

  describe "#save_model and #load_model" do
    it "saves and loads a trained model" do
      lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 500)

      features = [
        [0.0, 0.0],
        [10.0, 10.0],
      ]
      labels = ["a", "b"]

      lr.train(features, labels)

      # Save
      tmp_file = File.join(Dir.tempdir, "lr_test.msgpack")
      lr.save_model(tmp_file)

      # Load
      loaded_lr = Cadmium::Classifier::Tabular::LogisticRegression.load_model(tmp_file)

      # Verify predictions match
      loaded_lr.classify([0.0, 0.0]).should eq("a")
      loaded_lr.classify([10.0, 10.0]).should eq("b")

      # Verify weights match
      original_weights = lr.weights
      loaded_weights = loaded_lr.weights

      original_weights.size.should eq(loaded_weights.size)
      original_weights.each_with_index do |w, i|
        w.should be_close(loaded_weights[i], 1e-6)
      end

      lr.bias.should be_close(loaded_lr.bias, 1e-6)

      # Clean up
      File.delete(tmp_file)
    end
  end
end
