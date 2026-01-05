require "msgpack"

module Cadmium
  module Classifier
    module Tabular
      # Internal struct for MessagePack serialization
      private struct KNNData
        include MessagePack::Serializable

        property k : Int32
        property distance_metric : String
        property features : Array(Array(Float64))
        property labels : Array(String)
        property n_features : Int32

        def initialize(@k : Int32, @distance_metric : String, @features : Array(Array(Float64)), @labels : Array(String), @n_features : Int32)
        end
      end

      # K-Nearest Neighbors classifier for multi-feature tabular data.
      #
      # This classifier stores all training data and makes predictions by finding
      # the k most similar training examples and taking a majority vote.
      #
      # ### Features
      #
      # - Handles numerical features of any dimension
      # - Supports multiple distance metrics (Euclidean, Manhattan, Cosine, etc.)
      # - No training phase - just data storage
      # - Suitable for small to medium datasets
      #
      # ### Example
      #
      # ```
      # classifier = Cadmium::Classifier::Tabular::KNN.new(k: 3)
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
      # # Predict new sample
      # result = classifier.classify([1.05, 2.05, 3.05])
      # # => "class_a"
      #
      # # Get detailed results with vote counts
      # details = classifier.classify_details([1.05, 2.05, 3.05])
      # # => {"class_a" => 3, "class_b" => 0}
      # ```
      class KNN
        # Number of neighbors to consider
        getter k : Int32

        # Distance metric to use for finding nearest neighbors
        getter distance_metric : DistanceMetric

        # Training data storage
        @feature_matrix : Tensor(Float64, CPU(Float64))
        @label_vector : Array(String)
        @n_samples : Int32
        @n_features : Int32

        def initialize(@k : Int32 = 5, @distance_metric : DistanceMetric = DistanceMetric::Euclidean)
          @feature_matrix = Tensor.new([0, 0]) { 0.0 }
          @label_vector = [] of String
          @n_samples = 0
          @n_features = 0
        end

        # Train the classifier by storing feature vectors and labels.
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
          @n_features = features[0].size
          features.each do |f|
            raise ArgumentError.new("All feature vectors must have the same dimension") if f.size != @n_features
          end

          @n_samples = features.size

          # Convert features to tensor for efficient computation
          # Build the tensor row by row
          @feature_matrix = Tensor.new([@n_samples, @n_features]) { |i| 0.0 }
          features.each_with_index do |feature_vector, i|
            feature_vector.each_with_index do |value, j|
              @feature_matrix[i, j] = value
            end
          end

          @label_vector = labels.dup
          self
        end

        # Train with a single sample.
        #
        # ```
        # classifier.train([1.0, 2.0, 3.0], "class_a")
        # ```
        def train(features : Array(Float64), label : String) : self
          if @n_samples == 0
            @n_features = features.size
            @feature_matrix = Tensor.new([1, @n_features]) { |i| features[i] }
            @label_vector = [label]
            @n_samples = 1
          else
            raise ArgumentError.new("Feature dimension mismatch") if features.size != @n_features

            # Expand the feature matrix
            old_matrix = @feature_matrix
            @feature_matrix = Tensor.new([@n_samples + 1, @n_features]) { |i| 0.0 }

            # Copy old data
            @n_samples.times do |i|
              @n_features.times do |j|
                @feature_matrix[i, j] = old_matrix[i, j]
              end
            end

            # Add new sample
            features.each_with_index do |value, j|
              @feature_matrix[@n_samples, j] = value
            end

            @label_vector << label
            @n_samples += 1
          end
          self
        end

        # Classify a new sample and return the predicted label.
        #
        # ```
        # classifier.classify([1.0, 2.0, 3.0]) # => "class_a"
        # ```
        def classify(features : Array(Float64)) : String
          raise ArgumentError.new("Classifier has not been trained") if @n_samples == 0
          raise ArgumentError.new("Feature dimension mismatch") if features.size != @n_features

          # Find k nearest neighbors
          neighbors = find_nearest_neighbors(features)

          # Majority vote
          vote_counts = Hash(String, Int32).new(0)
          neighbors.each do |idx|
            label = @label_vector[idx]
            vote_counts[label] = vote_counts.fetch(label, 0) + 1
          end

          # Return label with most votes
          vote_counts.max_by { |_, count| count }[0]
        end

        # Classify a new sample and return detailed vote counts.
        #
        # ```
        # details = classifier.classify_details([1.0, 2.0, 3.0])
        # # => {"class_a" => 3, "class_b" => 2}
        # ```
        def classify_details(features : Array(Float64)) : Hash(String, Int32)
          raise ArgumentError.new("Classifier has not been trained") if @n_samples == 0
          raise ArgumentError.new("Feature dimension mismatch") if features.size != @n_features

          neighbors = find_nearest_neighbors(features)

          vote_counts = Hash(String, Int32).new(0)
          neighbors.each do |idx|
            label = @label_vector[idx]
            vote_counts[label] = vote_counts.fetch(label, 0) + 1
          end

          vote_counts
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

        # Find the k nearest neighbors for a given sample.
        private def find_nearest_neighbors(features : Array(Float64)) : Array(Int32)
          query_point = Tensor.new([@n_features]) { |i| features[i] }

          # Calculate distances to all training points
          distances = Array(Tuple(Int32, Float64)).new(@n_samples)

          @n_samples.times do |i|
            training_point = @feature_matrix[i, ...]
            distance = DistanceCalculator.calculate(training_point, query_point, @distance_metric)
            distances << {i, distance}
          end

          # Sort by distance and take k nearest
          distances.sort_by { |(_, dist)| dist }[0...@k].map { |(idx, _)| idx }
        end

        # Save the trained model to a file.
        #
        # ```
        # classifier.save_model("knn_model.msgpack")
        # ```
        def save_model(path : String) : Nil
          data = KNNData.new(
            k: @k,
            distance_metric: @distance_metric.to_s,
            features: tensor_to_nested_array(@feature_matrix),
            labels: @label_vector,
            n_features: @n_features,
          )

          File.write(path, data.to_msgpack)
        end

        # Load a trained model from a file.
        #
        # ```
        # classifier = Cadmium::Classifier::Tabular::KNN.load_model("knn_model.msgpack")
        # ```
        def self.load_model(path : String) : self
          data = KNNData.from_msgpack(File.read(path))

          k = data.k
          distance_metric = DistanceMetric.parse(data.distance_metric)
          labels = data.labels
          n_features = data.n_features
          features_array = data.features

          # Reconstruct feature matrix
          n_samples = features_array.size
          feature_matrix = Tensor.new([n_samples, n_features]) { |i| 0.0 }

          features_array.each_with_index do |row, i|
            row.each_with_index do |value, j|
              feature_matrix[i, j] = value
            end
          end

          # Create classifier and set internal state using private setter
          classifier = new(k: k, distance_metric: distance_metric)
          classifier.set_loaded_data(feature_matrix, labels, n_samples, n_features)

          classifier
        end

        # Internal method to set data after loading from MessagePack
        protected def set_loaded_data(@feature_matrix : Tensor(Float64, CPU(Float64)), @label_vector : Array(String), @n_samples : Int32, @n_features : Int32)
        end

        # Convert a Tensor to a nested array for serialization.
        private def tensor_to_nested_array(tensor : Tensor(Float64, CPU(Float64))) : Array(Array(Float64))
          rows = tensor.shape[0]
          cols = tensor.shape[1]
          result = Array(Array(Float64)).new(rows)

          rows.times do |i|
            row = Array(Float64).new(cols)
            cols.times do |j|
              row << tensor[i, j].value
            end
            result << row
          end

          result
        end
      end
    end
  end
end
