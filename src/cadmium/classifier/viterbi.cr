require "./classifier"

module Cadmium
  module Classifier
    # Smoothing https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
    # Good-Turing estimate
    # Jelinek-Mercer smoothing (interpolation)
    # Katz smoothing (backoff)
    # Witten-Bell smoothing
    # Absolute discounting
    # Kneser-Ney smoothing

    class Viterbi < Base
      @raw_training_data : Hash(String, Hash(String, Int32)) # token_content => {tag => tag_count}
      @token_count = Hash(String, Int32)                     # Count of given token in the entire corpus
      @token_tag_count = Hash(Tuple(String), Int32)          # Count of token and tag occurring together.
      @observation_space : Set(String)                       # Set of unique tokens
      @state_space : Set(String)                             # Set of unique tags
      @initial_probabilities
      property ngrams_size : Int32 = 3
      @sequence_of_observations : Set(Hash(String, String)) # Set of ngrams goldlabeled
      @transition_matrix : Matrix(Float64)                  # q(s|u, v) : Transition probability defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
      @emission_matrix : Matrix(Float64)                    # e(x|s) : Emission probability defined as the probability of making an observation x given that the state was s.

      # Count the occurrence of n tags in a sequence
      # @ngram_counts = {}

      # Count of n-1 tags (prior) in a sequence
      # prior_ngrams_counts = {}

      # A trigram Hidden Markov Model can be defined using

      # A finite set of states. (ie grammatical tags for POS tagging)
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

        #   # Mapping of a word to set of tags it occurred with in the entire corpus
        #   self.word_to_tag = {}

        #   # Count of word and tag occurring together.
        #   self.word_tag_count = {}

        #   # Count of given word in the entire corpus
        #   self.tag_count = {}

        #   # Count the occurrence of 3 tags in a sequence
        #   self.trigram_counts = {}

        #   # Count of two tags (prior) in a sequence
        #   self.bigram_counts = {}

        #   # Set of all the tags in the corpus
        #   self.tags = set()

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
