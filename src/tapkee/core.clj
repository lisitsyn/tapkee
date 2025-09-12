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
   :download-zip "https://github.com/lisitsyn/tapkee/archive/master.zip"
   :download-targz "https://github.com/lisitsyn/tapkee/archive/master.tar.gz"})

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

(defn downloads-dropdown []
  (dropdown "downloads" "Download"
            [{:shortname "zip" :longname "as .zip" :href (:download-zip cdn-urls)}
             {:shortname "targz" :longname "as .tar.gz" :href (:download-targz cdn-urls)}]))

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
        src-id (str (:shortname example) "Src")]
    (modal (:shortname example)
           (:longname example)
           [:div
            [:div {:id dsc-id} 
             [:script (htmlize-markdown (str "#" dsc-id) (:description example))]]
            [:br]
            [:pre {:id src-id}
             [:script (load-sources (str "#" src-id) (:source example))]]]
           "")))



(defn index []
  (html5
    [:head
      (tag-manager-head)
      [:title (:title site-config)]
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
         (downloads-dropdown)
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
