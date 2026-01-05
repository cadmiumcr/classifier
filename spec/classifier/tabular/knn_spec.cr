require "../../spec_helper"

describe Cadmium::Classifier::Tabular::KNN do
  describe "#initialize" do
    it "initializes with default values" do
      knn = Cadmium::Classifier::Tabular::KNN.new
      knn.k.should eq(5)
      knn.distance_metric.should eq(Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)
    end

    it "initializes with custom values" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)
      knn.k.should eq(3)
      knn.distance_metric.should eq(Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)
    end
  end

  describe "#train" do
    it "trains with batch data" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
      features = [[1.0, 2.0], [3.0, 4.0]]
      labels = ["a", "b"]

      knn.train(features, labels)
      # Should not raise
    end

    it "trains with single sample" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
      knn.train([1.0, 2.0], "a")
      knn.train([3.0, 4.0], "b")

      result = knn.classify([1.1, 2.1])
      result.should eq("a")
    end

    it "raises on empty data" do
      knn = Cadmium::Classifier::Tabular::KNN.new
      expect_raises(ArgumentError, "Cannot train on empty data") do
        knn.train([] of Array(Float64), [] of String)
      end
    end

    it "raises on mismatched dimensions" do
      knn = Cadmium::Classifier::Tabular::KNN.new
      features = [[1.0, 2.0], [3.0, 4.0, 5.0]]
      labels = ["a", "b"]

      expect_raises(ArgumentError, "All feature vectors must have the same dimension") do
        knn.train(features, labels)
      end
    end

    it "raises on mismatched feature and label counts" do
      knn = Cadmium::Classifier::Tabular::KNN.new
      features = [[1.0, 2.0]]
      labels = ["a", "b"]

      expect_raises(ArgumentError, "Features and labels must have same length") do
        knn.train(features, labels)
      end
    end
  end

  describe "#classify" do
    it "classifies a simple binary case" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)

      # Two distinct clusters
      features = [
        [1.0, 1.0],
        [1.1, 1.1],
        [5.0, 5.0],
        [5.1, 5.1],
      ]
      labels = ["a", "a", "b", "b"]

      knn.train(features, labels)

      # Point near cluster "a"
      knn.classify([1.05, 1.05]).should eq("a")

      # Point near cluster "b"
      knn.classify([5.05, 5.05]).should eq("b")
    end

    it "uses majority voting with k=3" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 3)

      features = [
        [1.0, 1.0], # a
        [1.1, 1.1], # a
        [1.2, 1.2], # a
        [5.0, 5.0], # b
      ]
      labels = ["a", "a", "a", "b"]

      knn.train(features, labels)

      # Point closer to b, but surrounded by 3 a's
      knn.classify([1.5, 1.5]).should eq("a")
    end

    it "raises if not trained" do
      knn = Cadmium::Classifier::Tabular::KNN.new
      expect_raises(ArgumentError, "Classifier has not been trained") do
        knn.classify([1.0, 2.0])
      end
    end

    it "raises on feature dimension mismatch" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
      knn.train([[1.0, 2.0]], ["a"])

      expect_raises(ArgumentError, "Feature dimension mismatch") do
        knn.classify([1.0, 2.0, 3.0])
      end
    end
  end

  describe "#classify_details" do
    it "returns vote counts for each class" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 3)

      features = [
        [1.0, 1.0], # a
        [1.1, 1.1], # a
        [1.2, 1.2], # a
        [5.0, 5.0], # b
      ]
      labels = ["a", "a", "a", "b"]

      knn.train(features, labels)

      details = knn.classify_details([1.05, 1.05])
      details["a"].should eq(3)
      details.has_key?("b").should be_false
    end

    it "shows split votes" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 3)

      features = [
        [1.0, 1.0], # a
        [5.0, 5.0], # b
        [5.1, 5.1], # b
      ]
      labels = ["a", "b", "b"]

      knn.train(features, labels)

      # Point in the middle
      details = knn.classify_details([3.0, 3.0])
      details["a"].should eq(1)
      details["b"].should eq(2)
    end
  end

  describe "#classify_batch" do
    it "classifies multiple samples" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)

      features = [
        [1.0, 1.0],
        [5.0, 5.0],
      ]
      labels = ["a", "b"]

      knn.train(features, labels)

      results = knn.classify_batch([[1.1, 1.1], [5.1, 5.1]])
      results.should eq(["a", "b"])
    end
  end

  describe "#save_model and #load_model" do
    it "saves and loads a trained model" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)

      features = [
        [1.0, 1.0],
        [1.1, 1.1],
        [5.0, 5.0],
      ]
      labels = ["a", "a", "b"]

      knn.train(features, labels)

      # Save
      tmp_file = File.join(Dir.tempdir, "knn_test.msgpack")
      knn.save_model(tmp_file)

      # Load
      loaded_knn = Cadmium::Classifier::Tabular::KNN.load_model(tmp_file)

      # Verify predictions match
      loaded_knn.classify([1.05, 1.05]).should eq("a")
      loaded_knn.classify([5.05, 5.05]).should eq("b")

      # Clean up
      File.delete(tmp_file)
    end
  end

  describe "different distance metrics" do
    it "works with Manhattan distance" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)

      features = [[1.0, 1.0], [5.0, 5.0]]
      labels = ["a", "b"]

      knn.train(features, labels)
      knn.classify([1.1, 1.1]).should eq("a")
    end

    it "works with Chebyshev distance" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Chebyshev)

      features = [[1.0, 1.0], [5.0, 5.0]]
      labels = ["a", "b"]

      knn.train(features, labels)
      knn.classify([1.1, 1.1]).should eq("a")
    end

    it "works with Cosine distance" do
      knn = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Cosine)

      features = [[1.0, 1.0], [5.0, 5.0]]
      labels = ["a", "b"]

      knn.train(features, labels)
      # [1.1, 1.1] is more similar in direction to [1.0, 1.0] than to [5.0, 5.0]
      knn.classify([1.1, 1.1]).should eq("a")
    end
  end
end
