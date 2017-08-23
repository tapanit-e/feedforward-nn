'use strict';

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

	// when the number of hidden units increases, learning rate has to decrease
	// for instance, hiddenUnits > 100, alpha ~0.01
	this.alpha = NN.Helper.getOption(args, 'alpha', 0.5);
	this.momentum = NN.Helper.getOption(args, 'momentum', 0.1);
	
	this.numOfHiddenUnits = NN.Helper.getOption(args, 'hiddenUnits', 3);
	this.numOfInputs = NN.Helper.getOption(args, 'input', 2) + 1;
	this.numOfOutputs = NN.Helper.getOption(args, 'output', 1);
	
	this.isQ = NN.Helper.getOption(args, 'isQ', false);

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

		// for q-learning, the initial weights must be quite small random floats
		for (var j = 0; j < this.numOfHiddenUnits; j++)
			this.weightInputs[i][j] = this.isQ ? 0.01 * NN.Helper.getRandom(-2, 2) : NN.Helper.getRandom(-2, 2);

	}

	for (var i = 0; i < this.numOfHiddenUnits; i++) {

		var arr = [];
		this.weightOutputs.push(arr);

		for (var j = 0; j < this.numOfOutputs; j++)
			this.weightOutputs[i][j] = this.isQ ? 0.01 * NN.Helper.getRandom(-2, 2) : NN.Helper.getRandom(-2, 2);

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

// predict
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

// backpropagate with momentum
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

NN.QLearning = function(args) {

	this.epsilon = NN.Helper.getOption(args, 'epsilon', 0.1);
	this.gamma = NN.Helper.getOption(args, 'gamma', 0.5);
	this.nn = new NN.FeedForwardNN(args);

};

NN.QLearning.prototype.argMax = function(list) {

	var max = list[0],
	    index = 0;
	
	for (var i = 1; i < list.length; i++) {
	
		if (list[i] > max) {
			
			max = list[i];
			index = i;
		
		}
		
	}
			
	return index;

};

NN.QLearning.prototype.randomIndex = function(list) {

	return Math.floor(Math.random() * list.length);

};

// predict the next action based on the max q-value
NN.QLearning.prototype.act = function(state) {

	if (Math.random() < this.epsilon) {
	
		var action = this.randomIndex(state);
	
	} else {

		var actions = this.nn.forward(state);
		var action = this.argMax(actions);
	
	}
	
	this.previousState = this.currentState;
	this.previousAction = this.currentAction;
	this.currentAction = action;
	this.currentState = state;
	
	return action;

};

NN.QLearning.prototype.learn = function(reward) {

	if ('undefined' !== typeof this.currentReward) {

		var actions = this.nn.forward(this.currentState);
		var qValue = this.currentReward + this.gamma * this.nn.activationOutputs[this.argMax(actions)];
		var predicts = this.nn.forward(this.previousState);
	
		predicts[this.previousAction] = qValue;
	
		this.nn.backward(predicts);
	
	}
	
	this.currentReward = reward;

};

/* test mlp and q-learning */
(function() {

	var options = {};
	
	options.alpha = 0.5;
	options.momentum = 0.1;
	options.hiddenUnits = 10;
	options.input = 2;
	options.output = 4;
	options.gamma = 0.3;
	options.epsilon = 0.1;
	options.isQ = true;
	
	var x = 0;
	var y = 0;
	
	var destX = 0.7;
	var destY = 0.7;
	
	var q = new NN.QLearning(options);
	
	var totalReward = 0;
	
	for (var i = 0; i < 10000; i++) {
	
		var action = q.act([x, y]);
		var reward;
		
		var curX = x;
		var curY = y;
		
		switch (action) {
		
			case 0: x += 0.01; y += 0.01; break;
			case 1: x += 0.01; y -= 0.01; break;
			case 2: x -= 0.01; y += 0.01; break;
			case 3: x -= 0.01; y -= 0.01; break; 
		
		}
		
		if (Math.abs(x - destX) <= Math.abs(curX - destX) && Math.abs(y - destY) <= Math.abs(curY - destY))
			reward = 1;
		else
			reward = -1;
		
		totalReward += reward;
		
		q.learn(reward);
	
	}
	
	console.log(x + ' ' + y);
	
	console.log(totalReward);

})();

(function() {

	var options = {};
	
	options.alpa = 0.5;
	options.momentum = 0.1;
	options.hiddenUnits = 3;
	options.input = 2;
	options.output = 1;
	options.isQ = false;

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

})();
/* end test */


(function() {

	if ('undefined' !== typeof module &&
		module.exports) {
	
		module.exports = NN;	
		
	} else {
	
		window.mlp = NN;
	
	}

})();
