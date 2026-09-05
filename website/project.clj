(defproject tapkee "1.0.0-SNAPSHOT"
  :description "Tapkee library website"
  :url "http://tapkee.lisitsyn.me"
  :dependencies [[org.clojure/clojure "1.3.0"]
                 [hiccup "1.0.2"]]
  :main tapkee.core
  ;; Exclude static directory from compilation and resource processing
  :source-paths ["src"]
  :resource-paths ["resources"]
  :clean-targets ^{:protect false} ["target" "static"])
