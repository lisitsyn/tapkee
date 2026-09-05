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
