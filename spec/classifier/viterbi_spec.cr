require "../spec_helper"

describe Cadmium::Classifier::Viterbi do
  subject = Cadmium::Classifier::Viterbi

  describe "#initialize" do
    it "successfully initalizes all defaults" do
      classifier = subject.new
      classifier.tokenizer.should be_a(Cadmium::Tokenizer::Word)
      classifier.training_data.should eq(Array(Tuple(String, String)).new)
      classifier.token_count.should eq(Hash(String, Int32).new)
      classifier.token_label_count.should eq(Hash(Tuple(String), Int32).new)
      classifier.observation_space.should eq(Set(String).new)
      classifier.ngrams_size.should eq(3)
      classifier.state_space.should eq(Set(String).new)
      classifier.sequence_of_observations.should eq(Array(Array(Tuple(String, String))).new)
    end

    it "uses a custom tokenizer" do
      classifier = subject.new(tokenizer: Cadmium::Tokenizer::Aggressive.new(lang: :en))
      classifier.tokenizer.should be_a(Cadmium::Tokenizer::Aggressive)
    end

    it "uses a different ngrams size" do
      classifier = subject.new(ngrams_size: 5)
      classifier.ngrams_size.should eq(5)
    end

    describe "#train" do
      it "calculates and updates pre-model values" do
        data_1 = [
          {"they", "pronoun"},
          {"drink", "verb"},
          {"water", "verb"},

        ]
        data_2 = [
          {"they", "pronoun"},
          {"like", "verb"},
          {"having", "verb"},
          {"a", "determinant"},
          {"drink", "noun"},
          {"to", "adverb"},
          {"water", "verb"},
          {"down", "verb"},
        ]
        classifier = subject.new
        classifier.train(data_1)
        classifier.train(data_2)
        classifier.token_count.should eq({"they" => 2, "drink" => 2, "water" => 2, "like" => 1, "having" => 1, "a" => 1, "to" => 1, "down" => 1})
        classifier.token_label_count.should eq({ {"they", "pronoun"} => 2, {"drink", "verb"} => 1, {"water", "verb"} => 2, {"like", "verb"} => 1, {"having", "verb"} => 1, {"a", "determinant"} => 1, {"drink", "noun"} => 1, {"to", "adverb"} => 1, {"down", "verb"} => 1 })
        classifier.token_to_label.should eq({"they" => ["pronoun"], "drink" => ["verb", "noun"], "water" => ["verb"], "like" => ["verb"], "having" => ["verb"], "a" => ["determinant"], "to" => ["adverb"], "down" => ["verb"]})
        classifier.observation_space.should eq(Set{"they", "drink", "water", "like", "having", "a", "to", "down"})
        classifier.state_space.should eq(Set{"pronoun", "verb", "determinant", "noun", "adverb"})
        classifier.sequence_of_observations.should eq([[{"they", "pronoun"}, {"drink", "verb"}, {"water", "verb"}], [{"they", "pronoun"}, {"like", "verb"}, {"having", "verb"}], [{"a", "determinant"}, {"drink", "noun"}, {"to", "adverb"}], [{"water", "verb"}, {"down", "verb"}, {"", ""}]])
        classifier.sequence_of_prior_observations.should eq([[{"drink", "verb"}, {"they", "pronoun"}], [{"like", "verb"}, {"they", "pronoun"}], [{"drink", "noun"}, {"a", "determinant"}], [{"down", "verb"}, {"water", "verb"}]])
        classifier.ngram_label_count.should eq({["pronoun", "verb", "verb"] => 2, ["determinant", "noun", "adverb"] => 1, ["verb", "verb", ""] => 1})
        classifier.prior_ngram_label_count.should eq({["verb", "pronoun"] => 2, ["noun", "determinant"] => 1, ["verb", "verb"] => 1})
      end
    end
  end
end