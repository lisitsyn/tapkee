(ns tapkee.core
  (:use [hiccup.core :only (html)]
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
  [:script """
(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-KNTD3JMZ');
   """])

(defn tag-manager-body []
  [:noscript
   [:iframe {:src "https://www.googletagmanager.com/ns.html?id=GTM-KNTD3JMZ" :height 0 :width 0 :style "display:none;visibility:hidden"}]
  ])

(defn social-media-meta []
  (list
   ;; Open Graph meta tags
   [:meta {:property "og:title" :content "Tapkee - Efficient Dimension Reduction Library"}]
   [:meta {:property "og:description" :content "Tapkee is a C++ template library for dimension reduction with 18+ algorithms including t-SNE, Isomap, LLE, PCA and more. Fast, efficient, and easy to use."}]
   [:meta {:property "og:type" :content "website"}]
   [:meta {:property "og:url" :content "https://tapkee.lisitsyn.me"}]
   [:meta {:property "og:image" :content "https://tapkee.lisitsyn.me/img/tapkee-preview.png"}]
   [:meta {:property "og:image:width" :content "1200"}]
   [:meta {:property "og:image:height" :content "630"}]
   [:meta {:property "og:site_name" :content "Tapkee"}]
   [:meta {:property "og:locale" :content "en_US"}]
   
   ;; Twitter Card meta tags
   [:meta {:name "twitter:card" :content "summary_large_image"}]
   [:meta {:name "twitter:title" :content "Tapkee - Efficient Dimension Reduction Library"}]
   [:meta {:name "twitter:description" :content "C++ template library for dimension reduction with 18+ algorithms including t-SNE, Isomap, LLE, PCA and more."}]
   [:meta {:name "twitter:image" :content "https://tapkee.lisitsyn.me/img/tapkee-preview.png"}]
   [:meta {:name "twitter:image:alt" :content "Tapkee dimension reduction library visualization"}]
   
   ;; Additional meta tags
   [:meta {:name "description" :content "Tapkee is a C++ template library for dimension reduction with 18+ algorithms including t-SNE, Isomap, LLE, PCA and more. Fast, efficient, and easy to use."}]
   [:meta {:name "keywords" :content "dimension reduction, machine learning, C++, t-SNE, PCA, Isomap, LLE, data visualization, manifold learning"}]
   [:meta {:name "author" :content "Sergey Lisitsyn"}]
   ))

(defn navbar [& elements]
  [:nav {:class "navbar fixed-top navbar-expand-lg navbar-light bg-white shadow"}
   [:div {:class "container"}
    [:a {:class "navbar-brand" :href "#"} ""]
    [:button {:class "navbar-toggler" :type "button" :data-bs-toggle "collapse" 
              :data-bs-target "#navbarNav" :aria-controls "navbarNav" 
              :aria-expanded "false" :aria-label "Toggle navigation"}
     [:span {:class "navbar-toggler-icon"}]]
    [:div {:class "collapse navbar-collapse" :id "navbarNav"}
     [:ul {:class "navbar-nav mx-auto"}
      elements]]]])

(defn dropdown [id name elements]
  [:li {:class "nav-item dropdown"}
    [:a {:id id :href "#" :role "button" :class "nav-link dropdown-toggle" 
         :data-bs-toggle "dropdown" :aria-expanded "false"}
      name]
    [:ul {:class "dropdown-menu" :aria-labelledby id}
      (map (fn [dropdown-element] 
             (let [has-external-href (contains? dropdown-element :href)]
               [:li
                [:a (merge {:class "dropdown-item"}
                           (if has-external-href
                             {:href (:href dropdown-element) :target "_blank"}
                             {:href "#"
                              :data-bs-toggle "modal"
                              :data-bs-target (str "#" (:shortname dropdown-element))}))
                 (:longname dropdown-element)]
               ]))
            elements)
    ]
  ])

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
            [{:shortname "bench" :longname "Benchmarks" :href "http://lisitsyn.github.io/tapkee_jmlr_benchmarks"}]))

(defn header []
  [:header {:class "gradient-header py-5 text-left"}
    [:div {:class "container"}
    [:h1 "Tapkee"]
    [:p "an efficient dimension reduction library"]]])

(defn readme []
  [:div {:class "container py-5"}
   [:section 
    [:p {:id "readme"}
     [:script """
        jQuery.get('md/README.markdown',
                   function(data) {
                      var output = marked.parse(data);
                      $(\"#readme\").html(output);
                      $(\"#readme pre\").each(function() {
                        if (!$(this).parent().hasClass('code-block-container')) {
                          var container = $('<div class=\"code-block-container\"></div>');
                          var copyBtn = $('<button class=\"copy-code-btn\" onclick=\"copyToClipboard(this)\">Copy</button>');
                          $(this).wrap(container);
                          $(this).parent().append(copyBtn);
                        }
                      });
                   });
      """]]]])

(defn htmlize-markdown [id file]
  (str """
      $(document).ready(function () {
         jQuery.get('" file "', function (d) {
              $('"id"').html(marked.parse(d));
              $('"id"').each(function(i,e){
                MathJax.typeset([e]);
                // Enhance code blocks in markdown
                $(e).find('pre code').each(function() {
                  hljs.highlightElement(this);
                });
                $(e).find('pre').each(function() {
                  if (!$(this).parent().hasClass('code-block-container')) {
                    var container = $('<div class=\"code-block-container\"></div>');
                    var copyBtn = $('<button class=\"copy-code-btn\" onclick=\"copyToClipboard(this)\">Copy</button>');
                    $(this).wrap(container);
                    $(this).parent().append(copyBtn);
                  }
                });
              });
         });
      })"""))

(defn load-sources [id file]
  (str """
       $(document).ready(function () {
         jQuery.get('" file "', function (d) {
              $('"id"').text(d);
              $('"id"').each(function(i,e){
                hljs.highlightElement(e);
                // Wrap in container and add copy button
                var container = $('<div class=\"code-block-container\"></div>');
                var copyBtn = $('<button class=\"copy-code-btn\" onclick=\"copyToClipboard(this)\">Copy</button>');
                $(e).wrap(container);
                $(e).parent().append(copyBtn);
              });
         });
        })"""))


(defn script [& operations]
  (str operations))

(defn index []
  (html5
    [:head
      (tag-manager-head)
      [:title "Tapkee"]
      ;;;
      ;; Social media meta tags
      (social-media-meta)
      ;;;
      (include-css "//cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css")
      (include-css "//fonts.googleapis.com/css?family=Nunito")
      (include-css "//cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github.min.css")
      (include-css "css/styles.css")
      ;;;
      (include-js "//cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js") 
      (include-js "//cdn.jsdelivr.net/npm/d3@3.5.17/d3.min.js") 
      (include-js "//cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js")
      (include-js "//cdn.jsdelivr.net/npm/marked@16.2.1/lib/marked.umd.min.js")
      (include-js "//cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js")
      ;;;
      (mathjax-config)
      (include-js "//cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
      ;;;
      [:script """
        function copyToClipboard(button) {
          const codeBlock = button.parentElement.querySelector('pre code, pre');
          const text = codeBlock.textContent || codeBlock.innerText;
          
          if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(function() {
              button.textContent = 'Copied!';
              button.style.backgroundColor = '#28a745';
              setTimeout(function() {
                button.textContent = 'Copy';
                button.style.backgroundColor = '#6c757d';
              }, 2000);
            });
          } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            try {
              document.execCommand('copy');
              button.textContent = 'Copied!';
              button.style.backgroundColor = '#28a745';
              setTimeout(function() {
                button.textContent = 'Copy';
                button.style.backgroundColor = '#6c757d';
              }, 2000);
            } catch (err) {
              console.error('Failed to copy text: ', err);
            }
            
            document.body.removeChild(textArea);
          }
        }
      """]
    ]
    [:body
     (tag-manager-body)
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
                       (:longname " ")
                       [:div {:class "method-content"} 
                        [:div {:id (str (:shortname method) "Content")}]]
                       [:script (htmlize-markdown (str "#" (:shortname method) "Content") (:markdown method))]))
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
  ))

(defn -main []
  (println (index)))
