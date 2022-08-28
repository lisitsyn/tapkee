(ns tapkee.core
  (:use [compojure.core :only (defroutes GET)]
        [compojure.route :as route]
        [ring.adapter.jetty :as ring]
        [hiccup.core :only (html)]
        [hiccup.page :only (html5 include-css include-js)]))

(def all-methods [
  {:shortname "lle" :longname "Kernel Locally Linear Embedding" :markdown "md/lle.markdown"}
  {:shortname "npe" :longname "Neighborhood Preserving Embedding" :markdown "md/npe.markdown"}
  {:shortname "ltsa" :longname "Local Tangent Space Alignment" :markdown "md/ltsa.markdown"}
  {:shortname "lltsa" :longname "Linear Local Tangent Space Alignment" :markdown "md/lltsa.markdown"}
  {:shortname "le" :longname "Laplacian Eigenmaps" :markdown "md/le.markdown"}
  {:shortname "lpp" :longname "Locality Preserving Projections" :markdown "md/lpp.markdown"}
  {:shortname "dm" :longname "Diffusion Map" :markdown "md/dm.markdown"}
  {:shortname "isomap" :longname "Isomap" :markdown "md/isomap.markdown"}
  {:shortname "lisomap" :longname "Landmark Isomap" :markdown "md/isomap.markdown"}
  {:shortname "mds" :longname "Multidimensional Scaling" :markdown "md/mds.markdown"}
  {:shortname "lmds" :longname "Landmark Multidimensional Scaling" :markdown "md/mds.markdown"}
  {:shortname "spe" :longname "Stochastic Proximity Embedding" :markdown "md/spe.markdown"}
  {:shortname "kpca" :longname "Kernel PCA" :markdown "md/kpca.markdown"}
  {:shortname "pca" :longname "PCA" :markdown "md/pca.markdown"}
  {:shortname "rp" :longname "Random Projection" :markdown "md/rp.markdown"}
  {:shortname "fa" :longname "Factor Analysis" :markdown "md/fa.markdown"}
  {:shortname "tsne" :longname "t-SNE" :markdown "md/tsne.markdown"}
  {:shortname "bhsne" :longname "Barnes-Hut-SNE" :markdown "md/tsne.markdown"}
  ])

(def all-graphical-examples [
  {:shortname "promoters" :longname "Promoters embedding" :script "js/promoters.js" :description "code/promoters.md"}                        
  {:shortname "words" :longname "Words embedding" :script "js/words.js" :description "code/words.md"}                        
  {:shortname "cbcl" :longname "MIT-CBCL faces embedding" :script "js/cbcl.js" :description "code/cbcl.md"}                        
  {:shortname "mnist" :longname "MNIST digits embedding" :script "js/mnist.js" :description "code/mnist.md"}                        
  {:shortname "faces" :longname "Faces embedding" :script "js/faces.js" :description "code/faces.md"}                        
  ])

(def all-usage-examples [
  {:shortname "minimal" :longname "Minimal C++ example" :description "code/minimal.md" :source "code/minimal.cpp"}
  {:shortname "rna" :longname "RNA C++ example" :description "code/rna.md" :source "code/rna.cpp"}
  {:shortname "precomputed" :longname "Precomputed distance C++ example" :description "code/precomputed.md" :source "code/precomputed.cpp"}
  ])

(defn mathjax-config []
  [:script """
   MathJax = {
      inlineMath: [['$','$']],
      displayMath: [['$$','$$']],
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }
   };
  """])

(defn google-analytics []
  [:script """
    var gaJsHost = ((\"https:\" == document.location.protocol) ? \"https://ssl.\" : \"http://www.\");
    document.write(unescape(\"%3Cscript src='\" + gaJsHost + \"google-analytics.com/ga.js' type='text/javascript'%3E%3C/script%3E\"));
    try {
      var pageTracker = _gat._getTracker(\"UA-37814556-1\");
      pageTracker._trackPageview();                                                                                                                                                            } catch(err) {}
  """])

(defn forkme []
  [:a {:href "https://github.com/lisitsyn/tapkee"}
   [:img {:style "position: fixed; top: 0; right: 0; border: 0; z-index: 10000; margin: 0;" 
          :src "https://s3.amazonaws.com/github/ribbons/forkme_right_white_ffffff.png"
          :alt "Fork me on GitHub"}]])

(defn navbar [& elements]
  [:div {:class "navbar navbar-inverse navbar-fixed-top"}
   [:div {:class "navbar-inner"}
    [:div {:class "container"}
     [:a {:class "brand" :href "#"}]
     [:ul {:class "nav" :role "navigation"}
      elements]]]])

(defn dropdown [id name elements]
  [:li {:class "dropdown"}
    [:a {:id id :href "#" :role "button" :class "dropdown-toggle" :data-toggle "dropdown"}
      [:span name]
      [:b {:class "caret"}]]
    [:ul {:class "dropdown-menu" :role "menu" :aria-labelledby id}
      (concat (map (fn [dropdown-element] 
             [:li
              [:a {:tabindex "-1" 
                   :href (get dropdown-element :href (str "#" (get dropdown-element :shortname)))
                   :data-toggle "modal"}
               (get dropdown-element :longname)]
             ])
            elements))
    ]
  ])

(defn modal [id header description javascript]
  [:div {:id id :class "modal hide fade" :tabindex "-1" :role "dialog" 
         :aria-labelledby (str id "Label") :aria-hidden "true"}
      [:div {:class "modal-header"}
        [:button {:type "button" :class "close" :data-dismiss "modal" :aria-hidden "true"}
         "close"]
        [:h3 {:id (str id "Label")}
         header]]
      [:div {:class "modal-body"}
       description
       [:div {:align "center" :id (str id "Plot")}
        javascript]]
      ])

(defn techniques-dropdown []
  (dropdown "methods" "Dimension reduction techniques"
            all-methods))

(defn graphical-examples-dropdown []
  (dropdown "graph_examples" "Graphical examples"
            all-graphical-examples))

(defn code-examples-dropdown []
  (dropdown "code_examples" "Usage examples"
            all-usage-examples))

(defn downloads-dropdown []
  (dropdown "downloads" "Download"
            [{:shortname "zip" :longname "as .zip" :href "https://github.com/lisitsyn/tapkee/archive/master.zip"}
             {:shortname "targz" :longname "as .tar.gz" :href "https://github.com/lisitsyn/tapkee/archive/master.tar.gz"}]))

(defn more-dropdown []
  (dropdown "more" "More"
            [{:shortname "doxygen" :longname "API documentation" :href "doxygen/html/index.html"}
             {:shortname "travis" :longname "Tapkee on Travis" :href "https://travis-ci.org/lisitsyn/tapkee"}
             {:shortname "tests" :longname "Test coverage" :href "lcov/index.html"}
             {:shortname "bench" :longname "Benchmarks" :href "http://lisitsyn.github.io/tapkee_jmlr_benchmarks"}]))

(defn header []
  [:header {:class "jumbotron subhead"}
    [:div {:class "container"}
    [:h1 "Tapkee"]
    [:p "an efficient dimension reduction library"]]])

(defn readme []
  [:div {:class "container" :data-offset "15" :data-target ".nav-collapse"}
   [:section 
    [:p {:id "readme"}
     [:script """
        jQuery.get('md/README.markdown',
                   function(data) {
                      var output = markdown.toHTML(data);
                      $(\"#readme\").html(output);
                   });
      """]]]])

(defn footer []
  [:footer {:class "footer"}
   [:div {:class "container"}
    [:p
      [:small 
       "Hosted on " [:a {:href "https://www.heroku.com/"} "Heroku"] " with help of "
       [:a {:href "https://github.com/weavejester/compojure"} "Compojure"] ", "
       [:a {:href "https://github.com/weavejester/hiccup"} "Hiccup"]
       " and " [:a {:href "http://twitter.github.com/bootstrap/"} "Bootstrap"] " libraries " [:br] 
       "Developed by " [:a {:href "mailto://lisitsyn.s.o@gmail.com"} "Sergey Lisitsyn"] ". Dedicated to Oksana Bayda"
        ]]]])

(defn htmlize-markdown [id file]
  (str "jQuery.get('" file "',function(d){$('"id"').html(markdown.toHTML(d));MathJax.Hub.Queue([\"Typeset\",MathJax.Hub]);});"))

(defn load-sources [id file]
  (str "jQuery.get('" file "',function(d){$('"id"').text(d);$('"id"').each(function(i,e){hljs.highlightBlock(e)});});"))

(defn script [& operations]
  (str operations))

(defn index []
  (html5
    [:head
      [:title "Tapkee"]
      ;;;
      (include-css "css/bootstrap.min.css") (include-css "css/bootstrap-modal.css")
      (include-css "css/bootstrap-responsive.css")
      (include-css "css/tipsy.css") 
      (include-css "css/docs.css")
      (include-css "http://fonts.googleapis.com/css?family=Nunito") (include-css "css/override.css") 
      (include-css "http://yandex.st/highlightjs/7.3/styles/default.min.css")
      ;;;
      (include-js "js/d3.v3.min.js") (include-js "js/jquery-1.9.1.min.js") 
      (include-js "js/jquery.tipsy.js") (include-js "js/bootstrap.min.js") 
      (include-js "js/markdown.js") 
      (include-js "js/bootstrap-modalmanager.js") 
      (include-js "js/bootstrap-modal.js") 
      (include-js "//cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.6.0/build/highlight.min.js")
      (include-js "//cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
      (mathjax-config)
      ;;;
      (forkme)]
    [:body
     [:div {:class "page-container"}
      (navbar
         (techniques-dropdown)
         (code-examples-dropdown)
         (graphical-examples-dropdown)
         (downloads-dropdown)
         (more-dropdown))
      (header)
      (readme)]
      (concat (map (fn [method] 
                (modal (:shortname method) 
                       (:longname method)
                       ""
                       [:script (htmlize-markdown (str "#" (:shortname method)) (:markdown method))]))
              all-methods))
      (concat (map (fn [example]
                (modal (:shortname example)
                       (:longname example)
                       (let [dsc-id (str (:shortname example) "Dsc")]
                        [:div {:id dsc-id} 
                         [:script (htmlize-markdown (str "#" dsc-id) (:description example))]])
                       (include-js (:script example))))
              all-graphical-examples))
      (concat (map (fn [example]
                (modal (:shortname example)
                       (:longname example)
                       (let [dsc-id (str (:shortname example) "Dsc") src-id (str (:shortname example) "Src")]
                         (concat [
                            [:div {:id dsc-id} 
                              [:script (htmlize-markdown (str "#" dsc-id) (:description example))]][:br]
                            [:pre {:id (str src-id)}
                              [:script (str (load-sources (str "#" src-id) (:source example)))]]]))
                       ""))
              all-usage-examples))
    ]
    (footer)
    (google-analytics)))

(defroutes routes
  (GET "/" [] (index))
  (route/resources "/")
  (route/not-found "/"))

(defn -main []
  (let [port (Integer/parseInt (or (System/getenv "PORT") "8080"))]
    (run-jetty routes {:port port :join? false})))
