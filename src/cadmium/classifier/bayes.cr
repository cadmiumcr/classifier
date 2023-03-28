require "./classifier"

module Cadmium
  module Classifier
    # This is a native-bayes classifier which used Laplace Smoothing. It can
    # be trained to categorize sentences based on the words in that
    # sentence.
    #
    # Example:
    #
    # ```
    # classifier = Cadmium::Classifier::Bayes.new
    #
    # # Train some angry examples
    # classifier.train("omg I can't believe you would do that to me", "angry")
    # classifier.train("I hate you so much!", "angry")
    # classifier.train("Just go. I don't need this.", "angry")
    # classifier.train("You're so full of shit!", "angry")
    #
    # # Some happy ones
    # classifier.train("omg you're the best!", "happy")
    # classifier.train("I can't believe how happy you make me", "happy")
    # classifier.train("I love you so damn much!", "happy")
    # classifier.train("You're the best!", "happy")
    #
    # # And some indifferent ones
    # classifier.train("Idk, what do you think?", "indifferent")
    # classifier.train("yeah that's ok", "indifferent")
    # classifier.train("cool", "indifferent")
    # classifier.train("I guess we could do that", "indifferent")
    #
    # # Now let's test it on a sentence
    # classifier.classify("You shit head!")
    # # => "angry"
    #
    # puts classifier.classify("You're the best :)")
    # # => "happy"
    #
    # classifier.classify("idk, my bff jill?")
    # # => "indifferent"
    # ```
    class Bayes
      include JSON::Serializable
      include YAML::Serializable
      DEFAULT_TOKENIZER = Cadmium::Tokenizer::Word.new

      @[JSON::Field(ignore: true)]
      @[YAML::Field(ignore: true)]
      property tokenizer : Cadmium::Tokenizer::Base = DEFAULT_TOKENIZER
      # The words to learn from.
      getter vocabulary : Array(String)

      # The total number of words in the vocabulary
      getter vocabulary_size : Int32

      # Number of documents we have learned from.
      getter total_documents : Int32

      # Document frequency table for each of our categories.
      getter doc_count : Hash(String, Int32)

      # For each category, how many total words were
      # mapped to it.
      getter word_count : Hash(String, Int32)

      # Word frequency table for each category.
      getter word_frequency_count : Hash(String, Hash(String, Int32))

      # Category names
      getter categories : Array(String)

      def initialize(tokenizer = nil)
        @tokenizer = tokenizer if tokenizer
        @vocabulary = [] of String
        @vocabulary_size = 0
        @total_documents = 0
        @doc_count = {} of String => Int32
        @word_count = {} of String => Int32
        @word_frequency_count = {} of String => Hash(String, Int32)
        @categories = [] of String
      end

      # Intializes each of our data structure entities for this
      # new category and returns `self`.
      def initialize_category(name)
        unless categories.includes?(name)
          categories << name
          doc_count[name] = 0
          word_count[name] = 0
          word_frequency_count[name] = {} of String => Int32
        end
        self
      end

      # Train our native-bayes classifier by telling it what
      # `category` the train `text` corresponds to.
      def train(text, category)
        # Intialize the category if it hasn't already been
        # initialized.
        initialize_category(category)

        # Update our count of how many documents are mapped to
        # this category.
        @doc_count[category] += 1

        # Update the total number of documents we have learned
        # from.
        @total_documents += 1

        # Normalize the text into a word array.
        tokens = @tokenizer.tokenize(text)

        # Get a frequency count for each token in the text.
        freq_table = frequency_table(tokens)

        # Update our vocabulary and our word frequency count
        # for this category.
        freq_table.each do |token, frequency|
          # Add this word to our vocabulary if it isn't already
          # there.
          @vocabulary << token unless vocabulary.includes?(token)

          # Update the frequency information for this word in
          # this category.
          if !@word_frequency_count[category][token]?
            @word_frequency_count[category][token] = frequency
          else
            @word_frequency_count[category][token] += 1
          end

          # Update the count of all words we have seen mapped to
          # this category.
          @word_count[category] += frequency
        end

        @vocabulary_size = @vocabulary.size

        self
      end

      # Determines what category the `text` belongs to.
      def classify(text : String)
        tokens = tokenizer.tokenize(text)
        freq_table = frequency_table(tokens)

        # Iterate through our categories to calculate log probabilities
        log_probabilities = @categories.map do |category|
          # Calculate the overall probability of this category
          category_probability = @doc_count[category].to_f64 / @total_documents.to_f64

          # Take the LOG of the probability of this category
          log_probability = Math.log(category_probability)

          # Now we need to iterate through each word in our text and add the log probability of that word belonging to this category
          freq_table.each do |token, frequency|
            # Calculate the probability that this word belongs to this category
            token_prob = token_probability(token, category)

            # Add the log probability of this word belonging to this category to our running total
            log_probability += frequency * Math.log(token_prob)
          end

          {category, log_probability}
        end

        # Find the maximum log probability
        max_log_prob = log_probabilities.max_by { |prob| prob[1] }[1]

        # Calculate the log-sum-exp of all the probabilities
        log_sum_exp = log_probabilities.map { |prob| prob[1] }.reduce(0_f64) do |sum, prob|
          sum + Math.exp(prob - max_log_prob)
        end

        # Normalize the log probabilities and convert them to regular probabilities
        probabilities = log_probabilities.map do |prob|
          {prob[0], Math.exp(prob[1] - max_log_prob - Math.log(log_sum_exp))}
        end

        # Convert the probabilities to percentages
        percentages = probabilities.map do |prob|
          {prob[0], prob[1] * 100}
        end

        # Return the sorted percentages as a Hash
        Hash(String, Float64).new.tap do |hash|
          percentages.sort_by { |prob| prob[1] }.reverse_each do |prob|
            hash[prob[0]] = prob[1]
          end
        end
      end

      # Calculate the probaility that a `token` belongs to
      # a `category`.
      def token_probability(token, category)
        # How many times this word has occured in documents
        # mapped to this category.
        word_freq = @word_frequency_count[category][token]? || 0

        # What is the count of all words that have ever
        # been mapped to this category.
        word_count = @word_count[category]

        # Use Laplace Add-1 Smoothing equation
        (word_freq.to_f64 + 1_f64) / (word_count.to_f64 + @vocabulary_size.to_f64)
      end

      # Build a frequency hash map where
      # - the keys are the entries in `tokens`
      # - the values are the frequency of each entry in `tokens`
      def frequency_table(tokens)
        tokens.reduce({} of String => Int32) do |map, token|
          if map.has_key?(token)
            map[token] += 1
          else
            map[token] = 1
          end
          map
        end
      end
    end
  end
end
