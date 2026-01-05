require "../../spec_helper"

describe Cadmium::Classifier::Tabular::DistanceCalculator do
  describe ".calculate" do
    it "calculates Euclidean distance" do
      a = Tensor.from_array([1.0, 2.0, 3.0])
      b = Tensor.from_array([1.0, 2.0, 3.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)
      result.should be_close(0.0, 1e-10)
    end

    it "calculates Euclidean distance for different vectors" do
      a = Tensor.from_array([0.0, 0.0, 0.0])
      b = Tensor.from_array([3.0, 4.0, 0.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)
      result.should be_close(5.0, 1e-10)
    end

    it "calculates Manhattan distance" do
      a = Tensor.from_array([0.0, 0.0, 0.0])
      b = Tensor.from_array([3.0, 4.0, 0.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)
      result.should be_close(7.0, 1e-10)
    end

    it "calculates Chebyshev distance" do
      a = Tensor.from_array([0.0, 0.0, 0.0])
      b = Tensor.from_array([3.0, 4.0, 0.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Chebyshev)
      result.should be_close(4.0, 1e-10)
    end

    it "calculates cosine distance for identical vectors" do
      a = Tensor.from_array([1.0, 2.0, 3.0])
      b = Tensor.from_array([1.0, 2.0, 3.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
      result.should be_close(0.0, 1e-10)
    end

    it "calculates cosine distance for orthogonal vectors" do
      a = Tensor.from_array([1.0, 0.0, 0.0])
      b = Tensor.from_array([0.0, 1.0, 0.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
      result.should be_close(1.0, 1e-10)
    end

    it "calculates cosine distance for opposite vectors" do
      a = Tensor.from_array([1.0, 2.0, 3.0])
      b = Tensor.from_array([-1.0, -2.0, -3.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
      result.should be_close(2.0, 1e-10)
    end

    it "handles zero vectors for cosine distance" do
      a = Tensor.from_array([0.0, 0.0, 0.0])
      b = Tensor.from_array([1.0, 2.0, 3.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
      result.should eq(1.0)
    end

    it "handles both zero vectors for cosine distance" do
      a = Tensor.from_array([0.0, 0.0, 0.0])
      b = Tensor.from_array([0.0, 0.0, 0.0])
      result = Cadmium::Classifier::Tabular::DistanceCalculator.calculate(a, b, Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
      result.should eq(0.0)
    end
  end
end
