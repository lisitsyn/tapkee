d3.json("data/words.json", function(json) {
    var width  = 400,
        height = 400,
        margin = 50;

    var color = d3.scale.category10();

    var svg = d3.select("#wordsPlot").append("svg:svg")
                .attr("width", width)
                .attr("height", height);

    var x_extent = d3.extent(json.data, function(d) { return d.cx });
    var y_extent = d3.extent(json.data, function(d) { return d.cy });
    var x_scale = d3.scale.linear().range([margin, width-margin])
                          .domain(x_extent);
    var y_scale = d3.scale.linear().range([height-margin, margin])
                          .domain(y_extent);
    var node = svg.selectAll("circle")
                  .data(json.data)
                  .enter().append("svg:circle")
                  .attr("cx", function(d) { return x_scale(d.cx) })
                  .attr("cy", function(d) { return y_scale(d.cy) })
                  .attr("r", function(d) { return 3 })
                  .style("stroke", "black")
                  .style("fill", function(d,i) { return color(0) });

    $('svg circle').tipsy({
      gravity: 'w',
      html: true,
      title: function() {
        var d = this.__data__;
            string = d.string;
        return string;
      }
    });

});

