d3.json('data/cbcl.json', (json) => {
  const width = 400;
  const height = 400;
  const margin = 50;

  const color = d3.scale.category10();

  const svg = d3.select('#cbclPlot').append('svg:svg')
    .attr('width', width)
    .attr('height', height);

  const xExtent = d3.extent(json.data, (d) => d.cx);
  const yExtent = d3.extent(json.data, (d) => d.cy);
  const xScale = d3.scale.linear().range([margin, width - margin])
    .domain(xExtent);
  const yScale = d3.scale.linear().range([height - margin, margin])
    .domain(yExtent);
  
  const node = svg.selectAll('circle')
    .data(json.data)
    .enter().append('svg:circle')
    .attr('cx', (d) => xScale(d.cx))
    .attr('cy', (d) => yScale(d.cy))
    .attr('r', () => 3)
    .style('stroke', 'black')
    .style('fill', () => color(0));

  // Add Bootstrap tooltip data attributes to circles
  $('#cbclPlot svg circle').each(function() {
    const d = this.__data__;
    const string = d.fname;
    const img = `<img src="img/faces_transparent/${string}"/>`;
    
    $(this).attr({
      'data-bs-toggle': 'tooltip',
      'data-bs-placement': 'right',
      'data-bs-html': 'true',
      'title': img
    });
  });
  
  // Initialize Bootstrap tooltips with better performance
  // Only create instances for elements that don't already have them
  const container = document.getElementById('cbclPlot');
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

