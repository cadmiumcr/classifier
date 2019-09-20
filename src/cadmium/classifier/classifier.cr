require "json"
require "yaml"

module Cadmium
  module Classifier
    abstract class Base
      abstract def classify(text : String)
    end
  end
end
