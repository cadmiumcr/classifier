require "num"
require "msgpack"

module Cadmium
  module Classifier
    module Tabular
      # Internal struct for MessagePack serialization
      private struct LogisticRegressionData
        include MessagePack::Serializable

        property weights : Array(Array(Float64))
        property bias : Float64
        property label_classes : Array(String)
        property learning_rate : Float64

        def initialize(@weights : Array(Array(Float64)), @bias : Float64, @label_classes : Array(String), @learning_rate : Float64)
        end
      end

      # Logistic Regression classifier for binary classification.
      #
      # Uses gradient descent to learn weights that minimize binary cross-entropy loss.
      # Suitable for binary classification tasks with numerical features.
      #
      # ### Features
      #
      # - Fast prediction after training (O(1))
      # - Probabilistic output
      # - Works well with large datasets
      #
      # ### Example
      #
      # ```
      # classifier = Cadmium::Classifier::Tabular::LogisticRegression.new(learning_rate: 0.01, max_iterations: 1000)
      #
      # features = [
      #   [1.0, 2.0, 3.0],
      #   [1.1, 2.1, 3.1],
      #   [5.0, 6.0, 7.0],
      # ]
      # labels = ["class_a", "class_a", "class_b"]
      #
      # classifier.train(features, labels)
      #
      # result = classifier.classify([1.05, 2.05, 3.05])
      # # => "class_a"
      #
      # probs = classifier.classify_probabilities([1.05, 2.05, 3.05])
      # # => {"class_a" => 0.85, "class_b" => 0.15}
      # ```
      class LogisticRegression
        # Learning rate for gradient descent
        getter learning_rate : Float64

        # Maximum number of training iterations
        getter max_iterations : Int32

        # Model parameters
        @weights : Tensor(Float64, CPU(Float64))
        @bias : Float64

        # Label classes (should be exactly 2 for binary classification)
        @label_classes : Set(String)

        # Training data (kept for potential retraining)
        @feature_matrix : Tensor(Float64, CPU(Float64))?
        @label_vector : Array(String)?

        def initialize(@learning_rate : Float64 = 0.01, @max_iterations : Int32 = 1000)
          @weights = Tensor.new([0]) { 0.0 }
          @bias = 0.0
          @label_classes = Set(String).new
        end

        # Train the classifier using gradient descent.
        #
        # ```
        # features = [[1.0, 2.0], [3.0, 4.0]]
        # labels = ["a", "b"]
        # classifier.train(features, labels)
        # ```
        def train(features : Array(Array(Float64)), labels : Array(String)) : self
          raise ArgumentError.new("Features and labels must have same length") if features.size != labels.size
          raise ArgumentError.new("Cannot train on empty data") if features.empty?

          # Validate all feature vectors have the same dimension
          n_features = features[0].size
          features.each do |f|
            raise ArgumentError.new("All feature vectors must have the same dimension") if f.size != n_features
          end

          # Get unique labels (should be exactly 2 for binary classification)
          @label_classes = labels.to_set
          raise ArgumentError.new("Logistic regression requires exactly 2 classes") if @label_classes.size != 2

          n_samples = features.size

          # Convert features to tensor
          feature_matrix = Tensor.new([n_samples, n_features]) { |i| 0.0 }
          features.each_with_index do |feature_vector, i|
            feature_vector.each_with_index do |value, j|
              feature_matrix[i, j] = value
            end
          end

          # Convert labels to binary (0 or 1)
          # First label in sorted order becomes 0, second becomes 1
          sorted_labels = @label_classes.to_a.sort
          label_map = {sorted_labels[0] => 0, sorted_labels[1] => 1}
          y = Tensor.new([n_samples]) { |i| label_map[labels[i]].to_f64 }

          # Initialize weights and bias
          @weights = Tensor.new([n_features]) { 0.0 }
          @bias = 0.0

          # Gradient descent
          @max_iterations.times do |iteration|
            # Forward pass: compute predictions
            # Reshape weights to column vector for matmul
            weights_col = @weights.reshape([n_features, 1])
            logits = feature_matrix.matmul(weights_col).reshape([n_samples]) + @bias
            predictions = sigmoid(logits)

            # Compute gradients
            error = predictions - y

            # Gradient for weights: (1/n) * X^T * error
            # Reshape error to column vector for matmul
            error_col = error.reshape([n_samples, 1])
            weights_gradient = (feature_matrix.transpose.matmul(error_col)).reshape([n_features]) / n_samples.to_f64

            # Gradient for bias: (1/n) * sum(error)
            bias_gradient = error.sum / n_samples.to_f64

            # Update parameters
            @weights = @weights - weights_gradient * @learning_rate
            @bias = @bias - bias_gradient * @learning_rate

            # Optional: early stopping if converged
            break if weights_gradient.map { |w| w.abs }.sum < 1e-6
          end

          # Store training data for potential retraining
          @feature_matrix = feature_matrix
          @label_vector = labels.dup

          self
        end

        # Classify a new sample and return the predicted label.
        #
        # ```
        # classifier.classify([1.0, 2.0, 3.0]) # => "class_a"
        # ```
        def classify(features : Array(Float64)) : String
          raise ArgumentError.new("Classifier has not been trained") if @label_classes.empty?

          probs = classify_probabilities(features)
          probs.max_by { |_, prob| prob }[0]
        end

        # Classify a new sample and return probability scores for both classes.
        #
        # ```
        # probs = classifier.classify_probabilities([1.0, 2.0, 3.0])
        # # => {"class_a" => 0.85, "class_b" => 0.15}
        # ```
        def classify_probabilities(features : Array(Float64)) : Hash(String, Float64)
          raise ArgumentError.new("Classifier has not been trained") if @label_classes.empty?
          raise ArgumentError.new("Feature dimension mismatch") if features.size != @weights.size

          # Convert features to tensor
          x = Tensor.new([features.size]) { |i| features[i] }

          # Compute logit: w^T * x + b
          # Use dot product for vector-vector multiplication
          logit = (@weights * x).sum + @bias

          # Apply sigmoid to get probability of positive class
          prob_positive = sigmoid_scalar(logit)
          prob_negative = 1.0 - prob_positive

          # Map probabilities to labels
          sorted_labels = @label_classes.to_a.sort
          {
            sorted_labels[1] => prob_positive * 100.0, # Convert to percentage
            sorted_labels[0] => prob_negative * 100.0,
          }
        end

        # Classify multiple samples at once.
        #
        # ```
        # results = classifier.classify_batch([[1.0, 2.0], [3.0, 4.0]])
        # # => ["class_a", "class_b"]
        # ```
        def classify_batch(features_batch : Array(Array(Float64))) : Array(String)
          features_batch.map do |features|
            classify(features)
          end
        end

        # Get the learned weights (for inspection/debugging).
        def weights : Array(Float64)
          raise ArgumentError.new("Classifier has not been trained") if @label_classes.empty?
          @weights.to_a
        end

        # Get the learned bias term.
        def bias : Float64
          raise ArgumentError.new("Classifier has not been trained") if @label_classes.empty?
          @bias
        end

        # Sigmoid function: 1 / (1 + exp(-x))
        private def sigmoid(tensor : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
          # Apply sigmoid element-wise: 1 / (1 + exp(-x))
          tensor.map { |x| 1.0 / (1.0 + Math.exp(-x)) }
        end

        # Sigmoid function for scalar values
        private def sigmoid_scalar(x : Float64) : Float64
          # Clamp x to prevent overflow
          x = Math.max(-500.0, Math.min(500.0, x))
          1.0 / (1.0 + Math.exp(-x))
        end

        # Save the trained model to a file.
        #
        # ```
        # classifier.save_model("logistic_regression_model.msgpack")
        # ```
        def save_model(path : String) : Nil
          raise ArgumentError.new("Classifier has not been trained") if @label_classes.empty?

          # Convert weights tensor to nested array (as a 2D array for consistency)
          weights_array = [@weights.to_a]

          data = LogisticRegressionData.new(
            weights: weights_array,
            bias: @bias,
            label_classes: @label_classes.to_a,
            learning_rate: @learning_rate,
          )

          File.write(path, data.to_msgpack)
        end

        # Load a trained model from a file.
        #
        # ```
        # classifier = Cadmium::Classifier::Tabular::LogisticRegression.load_model("logistic_regression_model.msgpack")
        # ```
        def self.load_model(path : String) : self
          data = LogisticRegressionData.from_msgpack(File.read(path))

          # Reconstruct weights from nested array
          weights_array = data.weights[0] # Extract the 1D array
          n_features = weights_array.size
          weights = Tensor.new([n_features]) { |i| weights_array[i] }

          # Create classifier and set internal state
          classifier = new(learning_rate: data.learning_rate)
          classifier.set_loaded_data(weights, data.bias, data.label_classes)

          classifier
        end

        # Internal method to set data after loading from MessagePack
        protected def set_loaded_data(@weights : Tensor(Float64, CPU(Float64)), @bias : Float64, label_classes_array : Array(String))
          @label_classes = label_classes_array.to_set
        end
      end
    end
  end
end
