require "json"
require "yaml"

module Cadmium
  module Classifier
    abstract class Base
      include JSON::Serializable
      include YAML::Serializable

      DEFAULT_TOKENIZER = Cadmium::Tokenizer::Word.new

      @[JSON::Field(ignore: true)]
      @[YAML::Field(ignore: true)]
      property tokenizer : Cadmium::Tokenizer::Base = DEFAULT_TOKENIZER

      abstract def classify(text : String)
    end
  end
end
