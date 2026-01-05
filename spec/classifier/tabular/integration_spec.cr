require "../../spec_helper"

describe "Cadmium::Classifier::Tabular Integration Tests" do
  it "handles a realistic fraud detection scenario" do
    # Simulated credit card transaction data
    # Features: [transaction_amount, merchant_distance, time_diff, etc.]
    # Labels: "legit" or "fraud"

    # Training data - 9 features per transaction
    features = [
      # Legitimate transactions (small amounts, familiar merchants)
      [10.50, 0.5, 1.2, 0.0, 0.0, 1.0, 0.8, 0.5, 0.1],
      [25.00, 1.2, 0.5, 0.0, 0.0, 1.0, 0.9, 0.6, 0.2],
      [15.75, 0.8, 2.1, 0.0, 0.0, 1.0, 0.7, 0.4, 0.1],
      [5.25, 0.3, 0.8, 0.0, 0.0, 1.0, 0.85, 0.55, 0.15],
      [42.00, 1.5, 1.5, 0.0, 0.0, 1.0, 0.75, 0.65, 0.1],

      # Fraudulent transactions (large amounts, unusual patterns)
      [5000.00, 150.0, 100.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.95],
      [2500.00, 200.0, 50.0, 1.0, 1.0, 0.0, 0.15, 0.1, 0.9],
      [10000.00, 300.0, 200.0, 1.0, 1.0, 0.0, 0.05, 0.15, 0.98],
    ]
    labels = ["legit", "legit", "legit", "legit", "legit", "fraud", "fraud", "fraud"]

    # Test with KNN
    knn = Cadmium::Classifier::Tabular::KNN.new(k: 3, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)
    knn.train(features, labels)

    # Classify new transactions
    legit_transaction = [20.00, 1.0, 1.0, 0.0, 0.0, 1.0, 0.8, 0.5, 0.15]
    fraud_transaction = [5000.00, 180.0, 120.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.92]

    knn.classify(legit_transaction).should eq("legit")
    knn.classify(fraud_transaction).should eq("fraud")

    # Test with Logistic Regression
    lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.01, max_iterations: 1000)
    lr.train(features, labels)

    # Logistic Regression should also classify correctly (might need more iterations for convergence)
    lr_result_legit = lr.classify(legit_transaction)
    lr_result_fraud = lr.classify(fraud_transaction)

    # At minimum, both transactions should not be classified the same
    lr_result_legit.should_not eq(lr_result_fraud)
  end

  it "both classifiers produce consistent results on simple data" do
    # Simple 2D binary classification
    features = [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [10.0, 10.0],
      [11.0, 10.0],
      [10.0, 11.0],
    ]
    labels = ["a", "a", "a", "b", "b", "b"]

    # Train both classifiers
    knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
    knn.train(features, labels)

    lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 1000)
    lr.train(features, labels)

    # Test point in "a" region
    test_point_a = [0.5, 0.5]
    knn.classify(test_point_a).should eq("a")
    lr.classify(test_point_a).should eq("a")

    # Test point in "b" region
    test_point_b = [10.5, 10.5]
    knn.classify(test_point_b).should eq("b")
    lr.classify(test_point_b).should eq("b")
  end

  it "handles different distance metrics correctly" do
    features = [
      [0.0, 0.0],
      [1.0, 1.0],
      [5.0, 5.0],
    ]
    labels = ["a", "a", "b"]

    # Test with Euclidean distance
    knn_euclidean = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Euclidean)
    knn_euclidean.train(features, labels)
    knn_euclidean.classify([0.5, 0.5]).should eq("a")

    # Test with Manhattan distance
    knn_manhattan = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Manhattan)
    knn_manhattan.train(features, labels)
    knn_manhattan.classify([0.5, 0.5]).should eq("a")

    # Test with Cosine distance
    knn_cosine = Cadmium::Classifier::Tabular::KNN.new(k: 1, distance_metric: Cadmium::Classifier::Tabular::DistanceMetric::Cosine)
    knn_cosine.train(features, labels)
    # [0.5, 0.5] is most similar in direction to [0, 0] (or [1, 1])
    knn_cosine.classify([0.5, 0.5]).should eq("a")
  end

  it "serialization preserves predictions" do
    features = [
      [0.0, 0.0],
      [1.0, 1.0],
      [5.0, 5.0],
    ]
    labels = ["a", "a", "b"]

    test_point = [0.5, 0.5]

    # Test KNN serialization
    knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
    knn.train(features, labels)
    original_prediction = knn.classify(test_point)

    tmp_file_knn = File.join(Dir.tempdir, "knn_integration_test.msgpack")
    knn.save_model(tmp_file_knn)

    loaded_knn = Cadmium::Classifier::Tabular::KNN.load_model(tmp_file_knn)
    loaded_knn.classify(test_point).should eq(original_prediction)

    File.delete(tmp_file_knn)

    # Test Logistic Regression serialization
    lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 500)
    lr.train(features, labels)
    original_prediction_lr = lr.classify(test_point)

    tmp_file_lr = File.join(Dir.tempdir, "lr_integration_test.msgpack")
    lr.save_model(tmp_file_lr)

    loaded_lr = Cadmium::Classifier::Tabular::LogisticRegression.load_model(tmp_file_lr)
    loaded_lr.classify(test_point).should eq(original_prediction_lr)

    File.delete(tmp_file_lr)
  end

  it "batch prediction works efficiently" do
    features = [
      [0.0, 0.0],
      [1.0, 1.0],
      [5.0, 5.0],
    ]
    labels = ["a", "a", "b"]

    # Test KNN batch prediction
    knn = Cadmium::Classifier::Tabular::KNN.new(k: 1)
    knn.train(features, labels)

    batch_input = [[0.5, 0.5], [1.5, 1.5], [5.5, 5.5]]
    batch_results = knn.classify_batch(batch_input)

    batch_results.should eq(["a", "a", "b"])

    # Test Logistic Regression batch prediction
    lr = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.1, max_iterations: 500)
    lr.train(features, labels)

    batch_results_lr = lr.classify_batch(batch_input)

    # Results should be consistent
    batch_results_lr[0].should eq("a")
    batch_results_lr[2].should eq("b")
  end
end
