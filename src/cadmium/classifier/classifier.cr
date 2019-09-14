require "json"
require "yaml"

module Cadmium
  module Classifier
    class Base
      include JSON::Serializable
      include YAML::Serializable

      DEFAULT_TOKENIZER = Cadmium::Tokenizer::Word.new

      @[JSON::Field(ignore: true)]
      @[YAML::Field(ignore: true)]
      property tokenizer : Cadmium::Tokenizer::Base = DEFAULT_TOKENIZER
    end
  end
end
