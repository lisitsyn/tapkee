template=doc/template/method_template.html
mathjax_template=doc/template/mathjax.html

rm -rf doc/methods/html
mkdir doc/methods/html/
for method in doc/methods/*.markdown
do
	woext=${method%.*}
	methodname=${woext##*/}
	pandoc --from=markdown --to=html $method --strict --mathjax --template $template -H $mathjax_template >> doc/methods/html/$methodname.html
done;
