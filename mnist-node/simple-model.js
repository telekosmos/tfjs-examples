const tf = require('@tensorflow/tfjs');
const data = require('./data');

const LEARNING_RATE = 0.5;
const EPOCHS = 10;
const BATCH_SIZE = 100;

// declare the training data placeholders
// input x - for 28 x 28 pixels = 784
// let x = tf.placeholder(tf.float32, [None, 784]);
// now declare the output data placeholder/input data for output layer - 10 digits
// let y = tf.placeholder(tf.float32, [None, 10]);
let x = tf.input({
  batchShape: [null, 784],
  dtype: 'float32'
});
let y = tf.input({
  batchShape: [null, 10],
  dtype: 'float32'
});

// weights and biases for the hidden and output layer
// declare the weights connecting the input to the hidden layer
let W1 = tf.variable(tf.randomNormal([784, 300], 0, 0.03));
let b1 = tf.variable(tf.randomNormal([300]));
// and the weights connecting the hidden layer to the output layer
let W2 = tf.variable(tf.randomNormal([300, 10], 0, 0.03));
let b2 = tf.variable(tf.randomNormal([10]));

// Ops
// Output for the hidden layer 'by hand': b + x*W
let hiddenLayerVal = tf.add(tf.matMul(x, W1), b1);
hiddenLayerVal = tf.relu(hiddenLayer); // rectified linear unit as activation function

let outputLayerVal = tf.softmax(tf.add(tf.matMul(y, W2), b2));

// cost lost function implemented (handmade) as a cross entropy function
let clippedVal = tf.clipByValue(outputLayerVal, 1e-10, 0.999999999);
let crossEntropy = -tf.mean(tf.sum(outputLayerVal * tf.log(clippedVal) + (1 - outputLayerVal) * tf.log(1 - clippedVal)), 1);

// add an optimiser
let optimiser = tf.train.sgd(LEARNING_RATE);

// Train the model
// Get the training data and start training the model

