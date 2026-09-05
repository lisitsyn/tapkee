(ns tapkee.core
  (:use [hiccup.core :only (html)]
        [hiccup.page :only (html5 include-css include-js)]))

;; Configuration constants
(def site-config
  {:title "Tapkee - Efficient Dimension Reduction Library"
   :description "Tapkee is a C++ template library for dimension reduction with 18+ algorithms including t-SNE, Isomap, LLE, PCA and more. Fast, efficient, and easy to use."
   :short-description "C++ template library for dimension reduction with 18+ algorithms including t-SNE, Isomap, LLE, PCA and more."
   :url "https://tapkee.lisitsyn.me"
   :image "https://tapkee.lisitsyn.me/img/tapkee-preview.png"
   :author "Sergey Lisitsyn"
   :gtm-id "GTM-KNTD3JMZ"
   :github-url "https://github.com/lisitsyn/tapkee"
   :benchmark-url "http://lisitsyn.github.io/tapkee_jmlr_benchmarks"})

(def cdn-urls
  {:bootstrap-css "//cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
   :bootstrap-js "//cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
   :google-fonts "//fonts.googleapis.com/css?family=Nunito"
   :highlight-css "//cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github.min.css"
   :highlight-js "//cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js"
   :jquery "//cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"
   :d3 "//cdn.jsdelivr.net/npm/d3@3.5.17/d3.min.js"
   :marked "//cdn.jsdelivr.net/npm/marked@16.2.1/lib/marked.umd.min.js"
   :mathjax "//cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
   :pypi "https://pypi.org/project/tapkee/"
   :github-releases "https://github.com/lisitsyn/tapkee/releases/latest"})

(def local-resources
  {:clipboard-js "js/clipboard.js"
   :loaders-js "js/loaders.js"
   :styles-css "css/styles.css"})

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
  {:shortname "ms" :longname "Manifold Sculpting" :markdown "md/ms.markdown"}
  ])

(def all-graphical-examples [
  {:shortname "promoters" :longname "Promoters embedding" :script "js/promoters.js" :description "code/promoters.md"}
  {:shortname "words" :longname "Words embedding" :script "js/words.js" :description "code/words.md"}
  {:shortname "cbcl" :longname "MIT-CBCL faces embedding" :script "js/cbcl.js" :description "code/cbcl.md"}
  {:shortname "mnist" :longname "MNIST digits embedding" :script "js/mnist.js" :description "code/mnist.md"}
  {:shortname "faces" :longname "Faces embedding" :script "js/faces.js" :description "code/faces.md"}
  ])

(def all-usage-examples [
  {:shortname "minimal" :longname "Minimal example" :description "code/minimal.md"
   :sources [{:lang "C++" :id "cpp" :file "code/minimal.cpp"}
             {:lang "Python" :id "py" :file "code/minimal.py"}
             {:lang "R" :id "r" :file "code/minimal.r"}]}
  {:shortname "rna" :longname "RNA example" :description "code/rna.md"
   :sources [{:lang "C++" :id "cpp" :file "code/rna.cpp"}
             {:lang "Python" :id "py" :file "code/rna.py"}
             {:lang "R" :id "r" :file "code/rna.r"}]}
  {:shortname "precomputed" :longname "Precomputed distance example" :description "code/precomputed.md"
   :sources [{:lang "C++" :id "cpp" :file "code/precomputed.cpp"}
             {:lang "Python" :id "py" :file "code/precomputed.py"}
             {:lang "R" :id "r" :file "code/precomputed.r"}]}
  ])

(defn mathjax-config []
  [:script """
   MathJax = {
      tex: {
        inlineMath: [['$','$']],
        displayMath: [['$$','$$']]
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }
   };
  """])

(defn tag-manager-head []
  [:script (str "
(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','" (:gtm-id site-config) "');")])

(defn tag-manager-body []
  [:noscript
   [:iframe {:src (str "https://www.googletagmanager.com/ns.html?id=" (:gtm-id site-config))
             :height 0 :width 0 :style "display:none;visibility:hidden"}]])

(defn favicon-and-manifest-meta []
  (list
   ;; Favicon and manifest - Safari-compatible order and configuration
   [:link {:rel "shortcut icon" :type "image/x-icon" :href "/img/favicon.ico"}]
   [:link {:rel "icon" :type "image/x-icon" :href "/img/favicon.ico"}]
   [:link {:rel "icon" :type "image/png" :sizes "16x16" :href "/img/favicon-16x16.png"}]
   [:link {:rel "icon" :type "image/png" :sizes "32x32" :href "/img/favicon-32x32.png"}]
   [:link {:rel "apple-touch-icon" :sizes "180x180" :href "/img/apple-touch-icon.png"}]
   [:link {:rel "apple-touch-icon-precomposed" :sizes "180x180" :href "/img/apple-touch-icon.png"}]
   [:meta {:name "msapplication-TileImage" :content "/img/apple-touch-icon.png"}]
   [:meta {:name "msapplication-TileColor" :content "#ffffff"}]
   [:link {:rel "manifest" :href "/manifest.json"}]))

(defn social-media-meta []
  (let [{:keys [title description short-description url image author]} site-config]
    (list
     ;; Open Graph meta tags
     [:meta {:property "og:title" :content title}]
     [:meta {:property "og:description" :content description}]
     [:meta {:property "og:type" :content "website"}]
     [:meta {:property "og:url" :content url}]
     [:meta {:property "og:image" :content image}]
     [:meta {:property "og:image:width" :content "1200"}]
     [:meta {:property "og:image:height" :content "630"}]
     [:meta {:property "og:site_name" :content "Tapkee"}]
     [:meta {:property "og:locale" :content "en_US"}]

     ;; Twitter Card meta tags
     [:meta {:name "twitter:card" :content "summary_large_image"}]
     [:meta {:name "twitter:title" :content title}]
     [:meta {:name "twitter:description" :content short-description}]
     [:meta {:name "twitter:image" :content image}]
     [:meta {:name "twitter:image:alt" :content "Tapkee dimension reduction library visualization"}]

     ;; Additional meta tags
     [:meta {:name "description" :content description}]
     [:meta {:name "keywords" :content "dimension reduction, machine learning, C++, t-SNE, PCA, Isomap, LLE, data visualization, manifold learning"}]
     [:meta {:name "author" :content author}]
     )))

(defn navbar [& elements]
  [:nav {:class "navbar fixed-top navbar-expand-lg navbar-light bg-white shadow"}
   [:div {:class "container"}
    [:a {:class "navbar-brand" :href "#"} ""]
    [:div {:class "collapse navbar-collapse" :id "navbarNav"}
     [:ul {:class "navbar-nav mx-auto"}
      elements]]]])

(defn dropdown-link [element]
  (let [has-external-href (contains? element :href)]
    [:li
     [:a (merge {:class "dropdown-item"}
                (if has-external-href
                  {:href (:href element) :target "_blank"}
                  {:href "#"
                   :data-bs-toggle "modal"
                   :data-bs-target (str "#" (:shortname element))}))
      (:longname element)]]))

(defn dropdown [id name elements]
  [:li {:class "nav-item dropdown"}
   [:a {:id id :href "#" :role "button" :class "nav-link dropdown-toggle"
        :data-bs-toggle "dropdown" :aria-expanded "false"}
    name]
   [:ul {:class "dropdown-menu" :aria-labelledby id}
    (map dropdown-link elements)]])

(defn modal [id header description javascript]
  [:div {:id id :class "modal fade" :tabindex "-1" :role "dialog"
         :aria-labelledby (str id "Label") :aria-hidden "true"}
    [:div {:class "modal-dialog modal-lg"}
      [:div {:class "modal-content"}
        [:div {:class "modal-header"}
          [:h5 {:class "modal-title" :id (str id "Label")} header]
          [:button {:type "button" :class "btn-close" :data-bs-dismiss "modal" :aria-label "Close"}]]
        [:div {:class "modal-body"}
         description
         [:div {:class "text-center" :id (str id "Plot")}
          javascript]]]]])

(defn github-button []
  [:li {:class "nav-item"}
   [:a {:class "nav-link" :href (:github-url site-config) :target "_blank"} "GitHub"]])

(defn techniques-dropdown []
  (dropdown "methods" "Dimension reduction techniques"
            all-methods))

(defn graphical-examples-dropdown []
  (dropdown "graph_examples" "Graphical examples"
            all-graphical-examples))

(defn code-examples-dropdown []
  (dropdown "code_examples" "Usage examples"
            all-usage-examples))

(defn install-dropdown []
  (dropdown "install" "Install"
            [{:shortname "pypi" :longname "Python (PyPI)" :href (:pypi cdn-urls)}
             {:shortname "github-release" :longname "Latest GitHub release" :href (:github-releases cdn-urls)}]))

(defn more-dropdown []
  (dropdown "more" "More"
            [{:shortname "bench" :longname "Benchmarks" :href (:benchmark-url site-config)}]))

(defn header []
  [:header {:class "gradient-header py-5 text-left"}
    [:div {:class "container"}
    [:h1 "Tapkee"]
    [:p "an efficient dimension reduction library"]]])

(defn readme []
  [:div {:class "container py-5"}
   [:section
    [:div {:id "readme"}]
    [:script "loadReadmeContent();"]]])

(defn htmlize-markdown [id file]
  (str "loadMarkdownContent('" id "', '" file "');"))

(defn load-sources [id file]
  (str "loadSourceCode('" id "', '" file "');"))

;; Modal helper functions
(defn method-modal [method]
  (modal (:shortname method)
         (:longname method)
         [:div {:class "method-content"}
          [:div {:id (str (:shortname method) "Content")}
           [:script (htmlize-markdown (str "#" (:shortname method) "Content") (:markdown method))]]]
         ""))

(defn graphical-example-modal [example]
  (let [dsc-id (str (:shortname example) "Dsc")]
    (modal (:shortname example)
           (:longname example)
           [:div {:id dsc-id}
            [:script (htmlize-markdown (str "#" dsc-id) (:description example))]]
           (include-js (:script example)))))

(defn usage-example-modal [example]
  (let [dsc-id (str (:shortname example) "Dsc")
        base-id (:shortname example)
        sources (:sources example)]
    (modal base-id
           (:longname example)
           [:div
            [:div {:id dsc-id}
             [:script (htmlize-markdown (str "#" dsc-id) (:description example))]]
            [:br]
            [:ul {:class "nav nav-tabs" :id (str base-id "-tabs") :role "tablist"}
             (map-indexed
               (fn [idx source]
                 [:li {:class "nav-item" :role "presentation"}
                  [:button {:class (str "nav-link" (when (zero? idx) " active"))
                            :id (str base-id "-" (:id source) "-tab")
                            :data-bs-toggle "tab"
                            :data-bs-target (str "#" base-id "-" (:id source))
                            :type "button"
                            :role "tab"}
                   (:lang source)]])
               sources)]
            [:div {:class "tab-content pt-3" :id (str base-id "-tab-content")}
             (map-indexed
               (fn [idx source]
                 (let [src-id (str base-id "-" (:id source) "-src")]
                   [:div {:class (str "tab-pane fade" (when (zero? idx) " show active"))
                          :id (str base-id "-" (:id source))
                          :role "tabpanel"}
                    [:pre {:id src-id}
                     [:script (load-sources (str "#" src-id) (:file source))]]]))
               sources)]]
           "")))



(defn index []
  (html5
    [:head
      (tag-manager-head)
      [:title (:title site-config)]
      ;; Favicon and manifest meta tags
      (favicon-and-manifest-meta)
      ;; Social media meta tags
      (social-media-meta)
      ;; CSS includes
      (include-css (:bootstrap-css cdn-urls))
      (include-css (:google-fonts cdn-urls))
      (include-css (:highlight-css cdn-urls))
      (include-css (:styles-css local-resources))
      ;; JavaScript includes
      (include-js (:jquery cdn-urls))
      (include-js (:d3 cdn-urls))
      (include-js (:highlight-js cdn-urls))
      (include-js (:marked cdn-urls))
      (include-js (:bootstrap-js cdn-urls))
      ;; MathJax configuration and script
      (mathjax-config)
      (include-js (:mathjax cdn-urls))
      ;; Local scripts
      [:script {:src (:clipboard-js local-resources)}]
      [:script {:src (:loaders-js local-resources)}]
    ]
    [:body
     (tag-manager-body)
     [:div {:class "page-container"}
      (navbar
         (techniques-dropdown)
         (code-examples-dropdown)
         (graphical-examples-dropdown)
         (install-dropdown)
         (more-dropdown)
         (github-button)
      )
      (header)
      (readme)]
      ;; Method modals
      (map method-modal all-methods)
      ;; Graphical example modals
      (map graphical-example-modal all-graphical-examples)
      ;; Usage example modals
      (map usage-example-modal all-usage-examples)
    ]
  ))

(defn -main []
  (println (index)))
