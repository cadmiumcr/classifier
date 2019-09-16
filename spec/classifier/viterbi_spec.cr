require "../spec_helper"

describe Cadmium::Classifier::Viterbi do
  subject = Cadmium::Classifier::Viterbi

  describe "#initialize" do
    it "successfully initalizes all defaults" do
      classifier = subject.new
      classifier.tokenizer.should be_a(Cadmium::Tokenizer::Word)
      classifier.raw_training_data.should eq(Hash(String, Hash(String, Int32)))
      classifier.token_count.should eq(Hash(String, Int32))
      classifier.token_tag_count.should eq(Hash(Tuple(String), Int32))
      classifier.observation_space.should eq(Set(String))
      classifier.ngrams_size.should eq(3)
      classifier.state_space.should eq(Set(String))
      classifier.sequence_of_observations.should eq(Set(Hash(String, String)))
    end
  end
end
