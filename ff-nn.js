var NN = NN || {};

NN.Helper = {};

NN.Helper.getOption = function(obj, key, def) {

	if ('undefined' === typeof obj)
		return def;
		
	return 'undefined' === typeof obj[key] ? def : obj[key];

};

NN.Helper.tanh = function(arg) {

	return Math.tanh(arg);

};

NN.Helper.getRandom = function(min, max) {

	return Math.random() * (max - min + 1) + min;

};

NN.Helper.dtanh = function(arg) {

	return 1.0 - arg * arg;

};

NN.FeedForwardNN = function(args) {

	this.alpha = NN.Helper.getOption(args, 'alpha', 0.5);
	this.momentum = NN.Helper.getOption(args, 'momentum', 0.1);
	
	this.numOfHiddenUnits = NN.Helper.getOption(args, 'hiddenUnits', 3);
	this.numOfInputs = NN.Helper.getOption(args, 'input', 2) + 1;
	this.numOfOutputs = NN.Helper.getOption(args, 'output', 1);

	this.activationInputs = [],
	this.activationHiddens = [],
	this.activationOutputs = [],
	this.weightInputs = [],
	this.weightOutputs = [],
	this.changeInputs = [], 
	this.changeOutputs = [];

	this.__initNet();

};

NN.FeedForwardNN.prototype.__initNet = function() {

	for (var i = 0; i < this.numOfInputs; i++)
		this.activationInputs.push(1.0);

	for (var i = 0; i < this.hiddens; i++)
		this.activationHiddens.push(1.0);


	for (var i = 0; i < this.numOfOutputs; i++)
		this.activationOutputs.push(1.0);

	for (var i = 0; i < this.numOfInputs; i++) {

		var arr = [];
		this.weightInputs.push(arr);

		for (var j = 0; j < this.numOfHiddenUnits; j++)
			this.weightInputs[i][j] = NN.Helper.getRandom(-2, 2);

	}

	for (var i = 0; i < this.numOfHiddenUnits; i++) {

		var arr = [];
		this.weightOutputs.push(arr);

		for (var j = 0; j < this.numOfOutputs; j++)
			this.weightOutputs[i][j] = NN.Helper.getRandom(-2, 2);

	}

	for (var i = 0; i < this.numOfInputs; i++) {

		var arr = [];
		this.changeInputs.push(arr);

		for (var j = 0; j < this.numOfHiddenUnits; j++)
			this.changeInputs[i][j] = 0.0;

	}

	for (var i = 0; i < this.numOfHiddenUnits; i++) {

		var arr = [];
		this.changeOutputs.push(arr);

		for (var j = 0; j < this.numOfOutputs; j++)
			this.changeOutputs[i][j] = 0.0;

	}

};

NN.FeedForwardNN.prototype.forward = function(input) {

	for (var i = 0; i < input.length; i++)
		this.activationInputs[i] = input[i];

	for (var i = 0; i < this.numOfHiddenUnits; i++) {

		var sum = 0.0;

		for (var j = 0; j < this.numOfInputs; j++)
			sum += this.activationInputs[j] * this.weightInputs[j][i]; 

		this.activationHiddens[i] = NN.Helper.tanh(sum);

	}

	for (var i = 0; i < this.numOfOutputs; i++) {

		var sum = 0.0;

		for (var j = 0; j < this.numOfHiddenUnits; j++)
			sum += this.activationHiddens[j] * this.weightOutputs[j][i];
		
		this.activationOutputs[i] = NN.Helper.tanh(sum);
	
	}
	
	var ret = [];
	
	for (var i = 0; i < this.activationOutputs.length; i++)
		ret[i] = this.activationOutputs[i];

	return ret;

};

NN.FeedForwardNN.prototype.backward = function(outputs) {

	var outputDeltas = [],
	    err = 0.0;

	for (var i = 0; i < this.numOfOutputs; i++) {

		err = outputs[i] - this.activationOutputs[i];
		outputDeltas[i] = NN.Helper.dtanh(this.activationOutputs[i]) * err;

	}

	var hiddenDeltas = [];

	for (var i = 0; i < this.numOfHiddenUnits; i++) {

		err = 0.0;

		for (var j = 0; j < this.numOfOutputs; j++)
			err += outputDeltas[j] * this.weightOutputs[i][j];

		hiddenDeltas[i] = NN.Helper.dtanh(this.activationHiddens[i]) * err;

	}

	var change = 0.0;

	for (var i = 0; i < this.numOfHiddenUnits; i++) {
	
		for (var j = 0; j < this.numOfOutputs; j++) {

			change = outputDeltas[j] * this.activationHiddens[i];

			this.weightOutputs[i][j] = this.weightOutputs[i][j] + this.alpha * change  + this.momentum * this.changeOutputs[i][j];
			this.changeOutputs[i][j] = change;

		}
	
	}

	for (var i = 0; i < this.numOfInputs; i++) {
	
		for (var j = 0; j < this.numOfHiddenUnits; j++) {

			change = hiddenDeltas[j] * this.activationInputs[i];

			this.weightInputs[i][j] = this.weightInputs[i][j] + this.alpha * change + this.momentum  * this.changeInputs[i][j];
			this.changeInputs[i][j] = change;

		}

	}

	var retErr = 0.0;

	for (var i = 0; i < outputs.length; i++)
		retErr += 0.5 * ((outputs[i] - this.activationOutputs[i]) * (outputs[i] - this.activationOutputs[i])); 

	return retErr;

};

var xor = function() {

	var options = {};
	
	options.alpa = 0.5;
	options.momentum = 0.1;
	options.hiddenUnits = 3;
	options.input = 2;
	options.output = 1;

	var nn = new NN.FeedForwardNN(options);

	for (var i = 0; i < 5000; i++) {

		nn.forward([1, 1]);
		nn.backward([-1]);
	
		nn.forward([1, 0]);
		nn.backward([1]);
	
		nn.forward([0, 1]);
		nn.backward([1]);
	
		nn.forward([0, 0]);
		nn.backward([-1]);

	}

	console.log(nn.forward([0, 1]));
	console.log(nn.forward([1, 1]));
	console.log(nn.forward([1, 0]));
	console.log(nn.forward([0, 0]));	

};

xor();
