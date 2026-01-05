require "./tabular/distance_metrics"
require "./tabular/knn"
require "./tabular/logistic_regression"

module Cadmium
  module Classifier
    # Tabular data classifiers for multi-feature numerical classification.
    #
    # Unlike text classifiers (Bayes) or sequence classifiers (Viterbi),
    # tabular classifiers work with numerical feature vectors.
    #
    # ### Example
    #
    # ```
    # require "cadmium_classifier"
    #
    # # KNN for simple classification
    # knn = Cadmium::Classifier::Tabular::KNN.new(k: 3)
    #
    # features = [
    #   [1.0, 2.0, 3.0],
    #   [1.1, 2.1, 3.1],
    #   [5.0, 6.0, 7.0],
    # ]
    # labels = ["class_a", "class_a", "class_b"]
    #
    # knn.train(features, labels)
    # knn.classify([1.05, 2.05, 3.05]) # => "class_a"
    #
    # # Logistic Regression for probabilistic classification
    # lr = Cadmium::Classifier::Tabular::LogisticRegression.new
    # lr.train(features, labels)
    # lr.classify([1.05, 2.05, 3.05]) # => "class_a"
    # ```
    module Tabular
    end
  end
end
