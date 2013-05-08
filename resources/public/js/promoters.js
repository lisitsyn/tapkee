d3.json("data/promoters.json", function(json) {
    var width  = 400,
        height = 400,
        margin = 50;

    var color = d3.scale.category10();
    var letters = ['A', 'C', 'T', 'G'];

    var svg = d3.select("#promotersPlot").append("svg:svg")
                .attr("width", width)
                .attr("height", height);

    var x_extent = d3.extent(json.data, function(d) { return d.cx });
    var y_extent = d3.extent(json.data, function(d) { return d.cy });
    var gc_extent = d3.extent(json.data, function(d) { return d.gc });
    var x_scale = d3.scale.linear().range([margin, width-margin])
                          .domain(x_extent);
    var y_scale = d3.scale.linear().range([height-margin, margin])
                          .domain(y_extent);
    var gc_scale = d3.scale.linear().range([0,1])
                          .domain(gc_extent);
    var node = svg.selectAll("circle")
                  .data(json.data)
                  .enter().append("svg:circle")
                  .attr("cx", function(d) { return x_scale(d.cx) })
                  .attr("cy", function(d) { return y_scale(d.cy) })
                  .attr("r", function(d) { return 5 })
                  .style("stroke", "black")
                  .style("fill", function(d,i) { return d3.hsl(200,0.5,gc_scale(d.gc)) });

    $('svg circle').tipsy({
      gravity: 'w',
      html: true,
      title: function() {
        var d = this.__data__;
            string = d.string;
        var text = string.substring(0,6) + " ... " + string.substring(string.length-6);
        for (var i=0; i<4; i++) {
          text = text.replace(new RegExp(letters[i],"g"),'<span style="color: ' + color(i) + ';">' + letters[i] + '</span>');
        }
        return text;
      }
    });

});

