require "./classifier"
require "apatite"

module Cadmium
  module Classifier
    # Smoothing https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
    # Good-Turing estimate
    # Jelinek-Mercer smoothing (interpolation)
    # Katz smoothing (backoff)
    # Witten-Bell smoothing
    # Absolute discounting
    # Kneser-Ney smoothing <= This is the one we need to implement !

    # struct TrainingData
    #   property data : Hash(String, String) # token => label
    #   property unique_tokens : Set(String)
    #   property unique_labels : Set(String)
    #   property label_to_token_count : Hash(String, Hash(String, Int32))

    #   def all
    #     @data
    #   end

    #   def initialize(data : Hash(String, String))
    #     @data = data
    #     @unique_tokens = self.ordered_tokens.to_set
    #     @unique_labels = self.ordered_labels.to_set
    #     @label_to_token_count = @data.tally
    #   end

    #   def ordered_tokens
    #     @data.keys
    #   end

    #   def ordered_labels
    #     @data.values
    #   end
    # end

    class Viterbi < Base
      include Apatite
      getter training_data : Array(Tuple(String, String))           # [] of [token,label]
      getter token_count : Hash(String, Int32)                      # Count of given token in the entire corpus
      getter token_to_label : Hash(String, Array(String | Nil))     # Hash mapping each token to one or several labels
      getter token_label_count : Hash(Tuple(String, String), Int32) # Count of token and label occurring together.
      getter observation_space : Set(String)                        # Set of tokens
      getter state_space : Set(String)                              # Set of labels
      # property initial_probabilities
      getter ngrams_size : Int32
      getter sequence_of_observations : Array(Array(Tuple(String, String)))       # Array of ngrams goldlabeled
      getter sequence_of_prior_observations : Array(Array(Tuple(String, String))) # Array of (n-1)grams goldlabeled
      getter ngram_label_count : Hash(Array(String), Int32)
      getter prior_ngram_label_count : Hash(Array(String), Int32)

      # @transition_matrix : Matrix(Float64)                  # q(s|u, v) : Transition probability defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
      # @emission_matrix : Matrix(Float64)                    # e(x|s) : Emission probability defined as the probability of making an observation x given that the state was s.

      def initialize(tokenizer = DEFAULT_TOKENIZER, ngrams_size = 3)
        @tokenizer = tokenizer
        @training_data = Array(Tuple(String, String)).new
        @token_count = Hash(String, Int32).new
        @token_label_count = Hash(Tuple(String, String), Int32).new
        @token_to_label = Hash(String, Array(String | Nil)).new
        @observation_space = Set(String).new
        @state_space = Set(String).new
        # @initial_probabilities
        @ngrams_size = ngrams_size
        @sequence_of_observations = Array(Array(Tuple(String, String))).new
        @sequence_of_prior_observations = Array(Array(Tuple(String, String))).new
        @ngram_label_count = Hash(Array(String), Int32).new
        @prior_ngram_label_count = Hash(Array(String), Int32).new
        # @transition_matrix = Matrix(Float64).new
        # @emission_matrix = Matrix(Float64).new
      end

      def train(training_data : Array(Tuple(String, String)))
        @training_data += training_data
        @token_count = @training_data.map { |tuple| tuple[0] }.tally
        @token_label_count = @training_data.tally
        @observation_space = @training_data.map { |tuple| tuple[0] }.to_set
        @token_to_label = @observation_space.to_h { |token| {token, @token_label_count.keys.map { |tuple| tuple.skip(1) if tuple.includes?(token) }.flatten.compact!} }
        @state_space = @training_data.map { |tuple| tuple[1] }.to_set
        @sequence_of_observations = @training_data.in_groups_of(@ngrams_size, {"", ""})
        @sequence_of_prior_observations = @sequence_of_observations.map { |ngram| ngram.reverse.skip(1) }
        @ngram_label_count = @sequence_of_observations.map { |ngram| ngram.map { |tuple| tuple.last } }.tally
        @prior_ngram_label_count = @sequence_of_prior_observations.map { |ngram| ngram.map { |tuple| tuple.last } }.tally
      end

      # Count the occurrence of n labels in a sequence
      # @ngram_counts = {}

      # Count of n-1 labels (prior) in a sequence
      # prior_ngrams_counts = {}

      # A trigram Hidden Markov Model can be defined using

      # A finite set of states. (ie grammatical labels for POS labelging)
      # A sequence of observations. (ie a dataset of trigrams goldlabeled)
      # q(s|u, v) : Transition probability defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
      # e(x|s) : Emission probability defined as the probability of making an observation x given that the state was s.

      # q(s|u, v) = c(u, v, s) / c(u, v)

      # e(x|s) = c(s → x) / c(s)

      #   c(u, v, s) represents the trigram count of states u, v and s. Meaning it represents the number of times the three states u, v and s occurred together in that order in the training corpus.
      #   c(u, v) following along similar lines as that of the trigram count, this is the bigram count of states u and v given the training corpus.
      #   c(s → x) is the number of times in the training set that the state s and observation x are paired with each other. And finally,
      #   c(s) is the prior probability of an observation being labelled as the state s.

      def viterbi(observation_space, state_space, initial_probabilities, sequence_of_observations, transition_matrix, emission_matrix) # : Array
        observation_space = sequence_of_observations if observation_space.nil?
        t1, t2 = Vector.zero(2)
        states_count = state_space.size
        state_space.each_with_index do |_, i|
          t1[i, 0] = initial_probabilities[i] * emission_matrix[i, sequence_of_observations.first]
        end
        observation_space.each_with_index(1) do |_, j|
          state_space.each_with_index do |_, i|
            t1[i, j] = max(t1[states_count, j - 1] * transition_matrix * emission_matrix)
            t2[i, j] = arg_max(t1[states_count, j - 1] * transition_matrix) # argmax = enumerable.max_by
          end
        end
        z = arg_max(t1[states_count, t1.size -1]) # arg_max = enumerable.max_by
        hidden_state_sequence[t1.size -1] = state_space[z]

        [observation_space.size - 1..2].each do |j|
          hidden_state_sequence[j - 1] = state_space[t2[]]
        end
        hidden_state_sequence
      end

      #       Smoothing

      # The idea behind Smoothing is just this:

      # 1.    Discount — the existing probability values somewhat and
      # 2.    Reallocate — this probability to the zeroes

      def smoothing
      end

      def classify(text : String)
      end

      def to_transition_probabilities
      end

      def to_emission_probabilities
      end

      def train(training_data : Hash(String, Hash(String, Int64)))
        #   # will hold conditional frequency distribution for P(Wi|Ck)
        #   self.words_given_pos = {}

        #   # will hold conditional frequency distribution for P(Ci+2|Ci+1, Ci)
        #   self.pos3_given_pos2_and_pos1 = {}

        #   # A helper object that gives us access to parsed files' data, both test and train.
        #   self.parser = DataParser(corpus_files)

        #   # An iterator over KFoldCrossValidation logic.
        #   self.cycle_iterator = KFoldCrossValidation(ConditionalProbability.k_fold, self.parser.get_training_data())

        #   # Mapping of a word to set of labels it occurred with in the entire corpus
        #   self.word_to_label = {}

        #   # Count of word and label occurring together.
        #   self.word_label_count = {}

        #   # Count of given word in the entire corpus
        #   self.label_count = {}

        #   # Count the occurrence of 3 labels in a sequence
        #   self.trigram_counts = {}

        #   # Count of two labels (prior) in a sequence
        #   self.bigram_counts = {}

        #   # Set of all the labels in the corpus
        #   self.labels = set()

        #   # Set of all the words in the corpus
        #   self.words = set()

        #   """ Back-off Probabilities """
        #   self.transition_backoff = {}
        #   self.emission_backoff = {}

        #   """ Singleton counts """
        #   self.transition_singleton = {}
        #   self.emission_singleton = {}

        #   """ 1-count smoothed probabilities """
        #   self.transition_one_count = {}
        #   self.emission_smoothed = {}

        #   self.n = 0
      end

      #     # http://www.adeveloperdiary.com/data-science/machine-learning/implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/
      #   # a and b are possible states (hidden states). V is the input vector containing the visible symbols (ie : words)
      #     def viterbi(V, a, b, initial_distribution):
      #       T = V.shape[0] # Vector of steps
      #       M = a.shape[0] # Matrix of probable states

      #       omega = np.zeros((T, M))
      #       omega[0, :] = np.log(initial_distribution * b[:, V[0]])

      #       prev = np.zeros((T - 1, M))

      #       for t in range(1, T):
      #           for j in range(M):
      #               # Same as Forward Probability
      #               probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

      #               # This is our most probable state given previous state at time t (1)
      #               prev[t - 1, j] = np.argmax(probability)

      #               # This is the probability of the most probable state (2)
      #               omega[t, j] = np.max(probability)

      #       # Path Array
      #       S = np.zeros(T)

      #       # Find the most probable last hidden state
      #       last_state = np.argmax(omega[T - 1, :])

      #       S[0] = last_state

      #       backtrack_index = 1
      #       for i in range(T - 2, -1, -1):
      #           S[backtrack_index] = prev[i, int(last_state)]
      #           last_state = prev[i, int(last_state)]
      #           backtrack_index += 1

      #       # Flip the path array since we were backtracking
      #       S = np.flip(S, axis=0)

      #       # Convert numeric values to actual hidden states
      #       result = []
      #       for s in S:
      #           if s == 0:
      #               result.append("A")
      #           else:
      #               result.append("B")

      #       result

    end
  end
end
