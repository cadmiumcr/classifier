require "../spec_helper"

describe Cadmium::Classifier::Viterbi do
  subject = Cadmium::Classifier::Viterbi
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
  test = ["they", "like", "having", "a", "drink", "to", "water", "down"]

  describe "#initialize" do
    it "successfully initalizes all defaults" do
      classifier = subject.new
      classifier.training_data.should eq(Array(Tuple(String, String)).new)
      classifier.token_count.should eq(Hash(String, Int32).new)
      classifier.token_label_count.should eq(Hash(Tuple(String), Int32).new)
      classifier.observation_space.should eq(Set(String).new)
      classifier.ngrams_size.should eq(3)
      classifier.state_space.should eq(Set(String).new)
      classifier.sequence_of_observations.should eq(Array(Array(Tuple(String, String))).new)
    end

    it "uses a different ngrams size" do
      classifier = subject.new(ngrams_size: 5)
      classifier.ngrams_size.should eq(5)
    end

    describe "#train" do
      it "calculates and updates pre-model values" do
        classifier = subject.new
        classifier.train(data_1)
        classifier.train(data_2)
        classifier.token_count.should eq({"they" => 2, "drink" => 2, "water" => 2, "like" => 1, "having" => 1, "a" => 1, "to" => 1, "down" => 1})
        classifier.token_label_count.should eq({ {"they", "pronoun"} => 2, {"drink", "verb"} => 1, {"water", "verb"} => 2, {"like", "verb"} => 1, {"having", "verb"} => 1, {"a", "determinant"} => 1, {"drink", "noun"} => 1, {"to", "adverb"} => 1, {"down", "verb"} => 1 })
        classifier.observation_space.should eq(Set{"they", "drink", "water", "like", "having", "a", "to", "down"})
        classifier.state_space.should eq(Set{"pronoun", "verb", "determinant", "noun", "adverb"})
        classifier.sequence_of_ngrams.should eq([[{"they", "pronoun"}, {"drink", "verb"}, {"water", "verb"}], [{"they", "pronoun"}, {"like", "verb"}, {"having", "verb"}], [{"a", "determinant"}, {"drink", "noun"}, {"to", "adverb"}], [{"water", "verb"}, {"down", "verb"}, {"", ""}]])
        classifier.sequence_of_prior_ngrams.should eq([[{"they", "pronoun"}, {"drink", "verb"}], [{"they", "pronoun"}, {"like", "verb"}], [{"a", "determinant"}, {"drink", "noun"}], [{"water", "verb"}, {"down", "verb"}]])
        classifier.ngram_label_count.should eq({["pronoun", "verb", "verb"] => 2, ["determinant", "noun", "adverb"] => 1, ["verb", "verb", ""] => 1})
        classifier.prior_ngram_label_count.should eq({["pronoun", "verb"] => 2, ["determinant", "noun"] => 1, ["verb", "verb"] => 1})
        classifier.transition_matrix.to_a.should eq([[4.9987534030914936e-5, 0.9999999950024929, 4.9987534030914936e-5, 4.9987534030914936e-5, 4.9987534030914936e-5], [4.997513649825938e-5, 0.9999999950049714, 4.997513649825938e-5, 4.997513649825938e-5, 4.997513649825938e-5], [9.990059442742952e-5, 9.990059442742952e-5, 9.990059442742952e-5, 0.9999999800397422, 9.990059442742952e-5], [3.990434412806819e-5, 0.8778955708175001, 3.990434412806819e-5, 3.990434412806819e-5, 0.4788521295368182], [0.4472135954999579, 0.4472135954999579, 0.4472135954999579, 0.4472135954999579, 0.4472135954999579]])
        classifier.emission_matrix.to_a.should eq([[0.9999991258754902, 0.0004997496880936983, 0.0004997496880936983, 0.0004997496880936983, 0.0004997496880936983, 0.0004997496880936983, 0.0004997496880936983, 0.0004997496880936983], [0.00035328834710904246, 0.3536416354561514, 0.7069299825651938, 0.3536416354561514, 0.3536416354561514, 0.00035328834710904246, 0.00035328834710904246, 0.3536416354561514], [0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.9999965070078156, 0.0009989975094983174, 0.0009989975094983174], [0.0009989975094983172, 0.9999965070078155, 0.0009989975094983172, 0.0009989975094983172, 0.0009989975094983172, 0.0009989975094983172, 0.0009989975094983172, 0.0009989975094983172], [0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.0009989975094983174, 0.9999965070078156, 0.0009989975094983174]])
      end
    end
    describe "#save_model and #load_model" do
      it "should ouput a json model and load while getting the same data" do
        classifier = subject.new
        classifier.train(data_2)
        classifier.save_model
        classifier2 = subject.new
        classifier2.load_model
        classifier.observation_space.should eq(classifier2.observation_space)
      end
    end

    describe "#classify" do
      it "should classify" do
        classifier = subject.new
        classifier.train(data_2)
        classifier.classify(test).should eq({"they" => "adverb", "like" => "verb", "having" => "verb", "a" => "verb", "drink" => "noun", "to" => "adverb", "water" => "noun", "down" => "adverb"})
      end
    end
  end
end
