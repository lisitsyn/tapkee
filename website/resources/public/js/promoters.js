d3.json('data/promoters.json', (json) => {
  const width = 400;
  const height = 400;
  const margin = 50;

  const color = d3.scale.category10();
  const letters = ['A', 'C', 'T', 'G'];

  const svg = d3.select('#promotersPlot').append('svg:svg')
    .attr('width', width)
    .attr('height', height);

  const xExtent = d3.extent(json.data, (d) => d.cx);
  const yExtent = d3.extent(json.data, (d) => d.cy);
  const gcExtent = d3.extent(json.data, (d) => d.gc);
  const xScale = d3.scale.linear().range([margin, width - margin])
    .domain(xExtent);
  const yScale = d3.scale.linear().range([height - margin, margin])
    .domain(yExtent);
  const gcScale = d3.scale.linear().range([0, 1])
    .domain(gcExtent);
  
  const node = svg.selectAll('circle')
    .data(json.data)
    .enter().append('svg:circle')
    .attr('cx', (d) => xScale(d.cx))
    .attr('cy', (d) => yScale(d.cy))
    .attr('r', () => 5)
    .style('stroke', 'black')
    .style('fill', (d) => d3.hsl(200, 0.5, gcScale(d.gc)));

  // Add Bootstrap tooltip data attributes to circles
  $('#promotersPlot svg circle').each(function() {
    const d = this.__data__;
    const string = d.string;
    let text = `${string.substring(0, 6)} ... ${string.substring(string.length - 6)}`;
    
    for (let i = 0; i < 4; i++) {
      text = text.replace(new RegExp(letters[i], 'g'), `<span style="color: ${color(i)};">${letters[i]}</span>`);
    }
    
    $(this).attr({
      'data-bs-toggle': 'tooltip',
      'data-bs-placement': 'right',
      'data-bs-html': 'true',
      'title': text
    });
  });
  
  // Initialize Bootstrap tooltips with better performance
  // Only create instances for elements that don't already have them
  const container = document.getElementById('promotersPlot');
  const tooltipElements = container.querySelectorAll('[data-bs-toggle="tooltip"]');
  
  tooltipElements.forEach((element) => {
    if (!bootstrap.Tooltip.getInstance(element)) {
      new bootstrap.Tooltip(element, {
        boundary: container,
        sanitize: false // Since we control the content
      });
    }
  });
});

