// Markdown loading and processing utilities

function loadMarkdownContent(selector, markdownFile) {
    $(document).ready(function () {
        jQuery.get(markdownFile, function (data) {
            const output = marked.parse(data);
            $(selector).html(output);
            
            // Apply syntax highlighting
            $(selector).find('pre code').each(function() {
                hljs.highlightElement(this);
            });
            
            // Add copy buttons to code blocks
            addCopyButtonsToCodeBlocks(selector);
            
            // Process MathJax
            $(selector).each(function(i, element) {
                MathJax.typeset([element]);
            });
        });
    });
}

function loadSourceCode(selector, sourceFile) {
    $(document).ready(function () {
        jQuery.get(sourceFile, function (data) {
            $(selector).text(data);
            $(selector).each(function(i, element) {
                hljs.highlightElement(element);
                addCopyButtonsToCodeBlocks($(element).parent());
            });
        });
    });
}

function addCopyButtonsToCodeBlocks(container) {
    $(container).find('pre').each(function() {
        if (!$(this).parent().hasClass('code-block-container')) {
            const codeContainer = $('<div class="code-block-container"></div>');
            const copyBtn = $('<button class="copy-code-btn" onclick="copyToClipboard(this)">Copy</button>');
            $(this).wrap(codeContainer);
            $(this).parent().append(copyBtn);
        }
    });
}

// Load README content specifically
function loadReadmeContent() {
    jQuery.get('md/README.markdown', function(data) {
        const output = marked.parse(data);
        $("#readme").html(output);
        addCopyButtonsToCodeBlocks("#readme");
    });
}
