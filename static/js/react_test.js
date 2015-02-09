var Data = React.createClass({
  render: function() {

  	//Make a pythonesque zip function 
  	//turns [[0,1,2],[10,11,12]] ===> [[0,10], [1,11], [2,12]]
  	function zip(arrays) {
	    return arrays[0].map(function(_,i){
	        return arrays.map(function(array){return array[i]})
	    });
	}

  	countlist = zip([this.props.text[0], this.props.text[1]]).map(function(kv){
  		return (
  			<li>{kv[0]}     {kv[1]}</li>
  		);
  	});

    return (
      <div className="data">
        <h2 className="dataaward">
          {this.props.award}
        </h2>
        <ol>
        	{countlist}
        </ol>
      </div>
    );
  }
});

var DataList = React.createClass({
  render: function() {
  	var data_nodes = this.props.data.map(function(data){
  		return (
  			<Data award={data.award} text={data.text}>
  				{data.text}
  			</Data>
  		);
  	});
    return (
      <div className="DataList">
        {data_nodes}
      </div>
    );
  }
});

var DataBox = React.createClass({
  load_data: function() {
    $.ajax({
      url: this.props.url,
      dataType: 'json',
      success: function(data) {
        this.setState({data: data});
      }.bind(this),
      error: function(xhr, status, err) {
        console.error(this.props.url, status, err.toString());
      }.bind(this)
    });
  },
  getInitialState: function() {
    return {data: []};
  },
  componentDidMount: function() {
    this.load_data();
    setInterval(this.load_data, this.props.pollInterval);
  },
  render: function() {
    return (
      <div className="DataBox">
        <h1>Awards</h1>
        <DataList data={this.state.data} />
      </div>
    );
  }
});

React.render(
  <DataBox url="thedata.json" pollInterval={2000} />,
  document.getElementById('content')
);