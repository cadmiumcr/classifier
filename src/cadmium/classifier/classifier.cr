require "json"
require "yaml"

module Cadmium
  module Classifier
    abstract class Base
      include JSON::Serializable
      include YAML::Serializable

      abstract def classify(text : String)
    end
  end
end
