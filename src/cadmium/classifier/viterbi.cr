require "./classifier"

module Cadmium
  module Classifier
    # TODO : add Kneser-Ney smoothing
    # This is a Hidden Markov Model classifier which uses the Viterbi algorithm.
    # It is efficient in predicting a state given prior observations matched to states.
    # In NLP, it is often used to attribut POS tags to words of a text.
    # As such it is used by Cadmium::POSTagger.
    class Viterbi
      include Apatite
      getter training_data : Array(Tuple(String, String)) # [] of [token,label]
      # observation_space, state_space, initial_probabilities, sequence_of_observations, transition_matrix, emission_matrix
      getter token_count : Hash(String, Int32)                      # Count of given token in the entire corpus
      getter label_count : Hash(String, Int32)                      # Count of given label in the entire corpus
      getter token_label_count : Hash(Tuple(String, String), Int32) # Count of token and label occurring together.

      # property initial_probabilities
      getter ngrams_size : Int32
      getter sequence_of_ngrams : Array(Array(Tuple(String, String)))       # A sequence of observations. (ie a dataset of trigrams goldlabeled)# Array of ngrams goldlabeled
      getter sequence_of_prior_ngrams : Array(Array(Tuple(String, String))) # Array of (n-1)grams goldlabeled
      getter ngram_label_count : Hash(Array(String), Int32)                 #
      getter prior_ngram_label_count : Hash(Array(String), Int32)
      # model data

      getter observation_space : Set(String)     # A finite set of possible observations. (ie a dictionnary of words)
      getter state_space : Set(String)           # A finite set of states. (ie grammatical labels for POS labelling)
      getter transition_matrix : Matrix(Float64) # q(s|u, v) : Transition probability defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
      getter emission_matrix : Matrix(Float64)   # e(x|s) : Emission probability defined as the probability of making an observation x given that the state was s.
      getter epsilon : Float64                   # Insignificant small number.
      getter sequence_of_observations : Array(String)
      getter predicted_states : Array(String)
      getter lookup_table : Hash(String, Int32)

      def initialize(ngrams_size = 3)
        @training_data = Array(Tuple(String, String)).new
        @token_count = Hash(String, Int32).new
        @label_count = Hash(String, Int32).new
        @token_label_count = Hash(Tuple(String, String), Int32).new
        @observation_space = Set(String).new
        @state_space = Set(String).new
        @ngrams_size = ngrams_size
        @sequence_of_ngrams = Array(Array(Tuple(String, String))).new
        @sequence_of_prior_ngrams = Array(Array(Tuple(String, String))).new
        @ngram_label_count = Hash(Array(String), Int32).new
        @prior_ngram_label_count = Hash(Array(String), Int32).new
        @transition_matrix = Matrix(Float64).build(1) { 0.0 }
        @emission_matrix = Matrix(Float64).build(1) { 0.0 }
        @epsilon = 0.000001
        @sequence_of_observations = Array(String).new
        @predicted_states = Array(String).new
        @lookup_table = Hash(String, Int32).new
      end

      # q(s|u, v) : Transition probability defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
      # q(s|u, v) = c(u, v, s) / c(u, v)
      #   c(u, v, s) represents the ngram count of states u, v and s. Meaning it represents the number of times the n states u, v, ..., and s occurred together in that order in the training corpus.
      #   c(u, v) following along similar lines as that of the ngram count, this is the n-1gram count of states u and v given the training corpus.
      private def transition_probability(ngram_label_count, prior_ngram_label_count) : Float64
        (ngram_label_count + 0.001 / (prior_ngram_label_count + 0.001*@state_space.size)).to_f
      end

      # e(x|s) : Emission probability defined as the probability of making an observation x given that the state was s.
      # e(x|s) = c(s → x) / c(s)
      #   c(s → x) is the number of times in the training set that the state s and observation x are paired with each other.
      #   c(s) is the prior probability of an observation being labelled as the state s.
      private def emission_probability(token_label_count, label_count) : Float64
        ((token_label_count + 0.001) / (label_count + 0.001*@observation_space.to_a.size)).to_f
      end

      def train(training_data : Array(Tuple(String, String)))
        @training_data += training_data
        @token_count = @training_data.map { |tuple| tuple[0] }.tally
        @label_count = @training_data.map { |tuple| tuple[1] }.tally
        @token_label_count = @training_data.tally
        @observation_space = @training_data.map { |tuple| tuple[0] }.to_set
        @observation_space.to_a.each_with_index { |token, i| @lookup_table[token] = i }
        @state_space = @training_data.map { |tuple| tuple[1] }.to_set
        @sequence_of_ngrams = @training_data.in_groups_of(@ngrams_size, {"", ""})
        @sequence_of_prior_ngrams = @sequence_of_ngrams.map { |ngram| ngram[...@ngrams_size - 1] }
        @ngram_label_count = @sequence_of_ngrams.map { |ngram| ngram.map(&.last) }.tally
        @prior_ngram_label_count = @sequence_of_prior_ngrams.map { |ngram| ngram.map(&.last) }.tally
        @transition_matrix = Matrix(Float64).build(@state_space.size) { 0.0 }
        @emission_matrix = Matrix(Float64).build(@state_space.size, @observation_space.size) { 0.0 }
        # Construct the Transition matrix
        @state_space.each_with_index do |state_1, i|
          @state_space.each_with_index do |state_2, j|
            ngram_index = @ngram_label_count.keys.index(&.join.includes?(state_1 + state_2))
            @transition_matrix[i, j] = transition_probability(@ngram_label_count.values[ngram_index], prior_ngram_label_count.fetch([state_1, state_2], 0.0)) if ngram_index
            @transition_matrix[i, j] = 0.0001 if !ngram_index
          end
        end
        @transition_matrix = Matrix.rows(@transition_matrix.row_vectors.map(&.normalize.to_a))
        # Construct the Emission matrix
        @state_space.each_with_index do |label, i|
          @observation_space.each_with_index do |token, j|
            @emission_matrix[i, j] = emission_probability(@token_label_count.fetch({token, label}, 0.0), @label_count.fetch(label, 0.0))
          end
        end
        @emission_matrix = Matrix.rows(@emission_matrix.row_vectors.map(&.normalize.to_a))
      end

      def save_model(filename : String = "model.zip")
        File.touch(filename)
        File.open(filename, "w") do |file|
          Compress::Zip::Writer.open(file) do |zip|
            zip.add("observation-space.json", @observation_space.to_json)
            zip.add("state-space.json", @state_space.to_json)
            zip.add("transition-matrix.json", @transition_matrix.to_a.to_json)
            zip.add("emission-matrix.json", @emission_matrix.to_a.to_json)
          end
        end
      end

      def load_model(filename : String = "model.zip")
        File.open(filename) do |file|
          Compress::Zip::Reader.open(file) do |zip|
            zip.each_entry do |entry|
              @observation_space = Set(String).from_json(entry.io.gets_to_end) if entry.filename == "observation-space.json"
              @state_space = Set(String).from_json(entry.io.gets_to_end) if entry.filename == "state-space.json"
              @transition_matrix = Matrix(Float64).from_json(entry.io.gets_to_end) if entry.filename == "transition-matrix.json"
              @emission_matrix = Matrix(Float64).from_json(entry.io.gets_to_end) if entry.filename == "emission-matrix.json"
            end
          end
        end
        @lookup_table = Hash(String, Int32).new
        @observation_space.to_a.each_with_index { |token, i| @lookup_table[token] = i } # for performance reasons
      end

      def classify(sequence_of_observations : Array(String)) : Hash(String, String)
        @sequence_of_observations = sequence_of_observations
        @predicted_states = Array(String).new(@sequence_of_observations.size, "")
        t1 = Matrix(Float64).build(@state_space.size, @sequence_of_observations.size) { 0.0 }
        t2 = Matrix(Int32).build(@state_space.size, @sequence_of_observations.size) { 0 }

        # Calculates the initial probabilities

        @state_space.each_with_index do |_, i|
          if @transition_matrix[0, i] == 0.0
            t1[i, 0] = -1.7976931348623157e+308
            t2[i, 0] = 0
          else
            t1[i, 0] = Math.log(@transition_matrix[0, i]) + Math.log(@emission_matrix[i, @lookup_table.fetch(@sequence_of_observations.first, 0)])
            t2[i, 0] = 0
          end
        end

        @sequence_of_observations.each_with_index do |token, i|
          @state_space.each_with_index do |_, j|
            best_probability = -1.7976931348623157e+308
            best_path = 0
            @state_space.each_with_index do |_, k|
              probability = t1[k, i - 1] + Math.log(@transition_matrix[k, j]) + Math.log(@emission_matrix[j, @lookup_table.fetch(token, 0)])
              if probability > best_probability
                best_probability = probability
                best_path = k
              end
              t1[j, i] = best_probability
              t2[j, i] = best_path
            end
          end
        end

        predicted_values = Array(Int32).new(@sequence_of_observations.size, 0)
        argmax = t1[0, t1.column_count - 1]

        @state_space.to_a[1..].each_with_index do |_, k|
          if t1[k, t1.column_count - 1] > argmax
            argmax = t1[k, t1.column_count - 1]
            predicted_values[@sequence_of_observations.size - 1] = k
          end
        end

        @predicted_states[@sequence_of_observations.size - 1] = @state_space.to_a[predicted_values[@sequence_of_observations.size - 1]]

        @sequence_of_observations.each_with_index do |_, i|
          predicted_values[i - 1] = t2[predicted_values[i], i]
          @predicted_states[i - 1] = @state_space.to_a[predicted_values[i - 1]]
        end
        @sequence_of_observations.zip(@predicted_states).to_h
      end
    end
  end
end
