require "num"

module Cadmium
  module Classifier
    module Tabular
      # Distance metrics for calculating similarity between feature vectors.
      #
      # Used primarily by KNN to find nearest neighbors.
      enum DistanceMetric
        # Euclidean distance: √Σ(aᵢ - bᵢ)²
        # Most common distance metric, works well for most cases
        Euclidean

        # Manhattan distance: Σ|aᵢ - bᵢ|
        # Also known as L1 distance or city block distance
        # Less sensitive to outliers than Euclidean
        Manhattan

        # Chebyshev distance: max|aᵢ - bᵢ|
        # Also known as L∞ distance or chessboard distance
        # Useful for grid-like data
        Chebyshev

        # Cosine distance: 1 - (a·b)/(||a||·||b||)
        # Measures angular similarity, ignores magnitude
        # Useful for high-dimensional data
        Cosine
      end

      # Calculates distances between feature vectors using various metrics.
      #
      # All methods accept num.cr Tensors for efficient vectorized operations.
      class DistanceCalculator
        # Calculate the distance between two vectors using the specified metric.
        #
        # ```
        # a = Tensor.from_array([1.0, 2.0, 3.0])
        # b = Tensor.from_array([1.1, 2.1, 3.1])
        #
        # DistanceCalculator.calculate(a, b, DistanceMetric::Euclidean)
        # # => 0.1732...
        # ```
        def self.calculate(a : Tensor(Float64, CPU(Float64)),
                           b : Tensor(Float64, CPU(Float64)),
                           metric : DistanceMetric) : Float64
          case metric
          when DistanceMetric::Euclidean
            euclidean_distance(a, b)
          when DistanceMetric::Manhattan
            manhattan_distance(a, b)
          when DistanceMetric::Chebyshev
            chebyshev_distance(a, b)
          when DistanceMetric::Cosine
            cosine_distance(a, b)
          else
            raise ArgumentError.new("Unsupported distance metric: #{metric}")
          end
        end

        # Calculate Euclidean distance: √Σ(aᵢ - bᵢ)²
        private def self.euclidean_distance(a, b)
          diff = a - b
          Math.sqrt((diff * diff).sum)
        end

        # Calculate Manhattan distance: Σ|aᵢ - bᵢ|
        private def self.manhattan_distance(a, b)
          diff = a - b
          diff.map { |i| i.abs }.sum
        end

        # Calculate Chebyshev distance: max|aᵢ - bᵢ|
        private def self.chebyshev_distance(a, b)
          diff = a - b
          diff.map { |i| i.abs }.max
        end

        # Calculate cosine distance: 1 - cosine_similarity
        # Cosine similarity = (a·b) / (||a|| * ||b||)
        private def self.cosine_distance(a, b)
          dot_product = (a * b).sum
          norm_a = Math.sqrt((a * a).sum)
          norm_b = Math.sqrt((b * b).sum)

          # Handle zero vectors
          return 0.0 if norm_a == 0.0 && norm_b == 0.0
          return 1.0 if norm_a == 0.0 || norm_b == 0.0

          cosine_sim = dot_product / (norm_a * norm_b)
          # Clamp to handle numerical errors
          cosine_sim = Math.max(-1.0, Math.min(1.0, cosine_sim))
          1.0 - cosine_sim
        end
      end
    end
  end
end
