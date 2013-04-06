This example shows how Tapkee can be used with Python, thanks to the language
bindings provided by Shogun. The data consists of words from the English
vocabulary. These words belong to different grammar groups: there are several
common nouns (e.g. cowboy, dragon), adjectives (e.g. harmful), proper nouns such
as Rivera or America, verbs in different forms (e.g. in gerund such as finishing
or in past participle like disrupted), among other classes. The method used for
embedding is Kernel Locally Linear Embedding (KLLE) and the callback required
in this case is a kernel callback. The kernel callback is defined in Python, in
the function word_kernel. Note the power of using language bindings
and how code written in target interfaces (Python in this case) is able to interact
with the underlying C++ code. The word kernel used is rather simple and it is
based on a measure of sequence similarity implemented in the difflib module of
Python (import the difflib module and issue help(difflib.SequenceMatcher)
from a Python console for more information, tested in both Python 2.7.3 and
Python 3.3.0). The target dimension of the embedding is two and, as it can be
seen in the figure, the different word classes form clusters in the two dimensional
plane.
