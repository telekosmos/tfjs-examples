const tf = require('@tensorflow/tfjs');

// Fit a quadratic function by learning the coefficients a, b, c.
const xs = tf.tensor1d([0, 1, 2, 3]);
const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);

const a = tf.scalar(Math.random()).variable();
const b = tf.scalar(Math.random()).variable();
const c = tf.scalar(Math.random()).variable();

// y = a * x^2 + b * x + c.
const f = x => a.mul(x.square()).add(b.mul(x)).add(c); // layer
const loss = (pred, label) => pred.sub(label).square().mean();

const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

// Train the model.
for (let i = 0; i < 10; i++) {
  optimizer.minimize(() => loss(f(xs), ys));
}

// Make predictions.
/*
console.log(`a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
  console.log(`x: ${i}, pred: ${pred}`);
});
*/

///////////////////////////////////////////////////////////////////////
// GENERIC MODEL
// Define input, which has a size of 5 (not including batch dimension).
const input = tf.input({ shape: [5] });

// First dense layer uses relu activation.
const denseLayer1 = tf.layers.dense({ units: 10, activation: 'relu' });
// Second dense layer uses softmax activation.
const denseLayer2 = tf.layers.dense({ units: 2, activation: 'softmax' });

// Obtain the output symbolic tensor by applying the layers on the input.
const output = denseLayer2.apply(denseLayer1.apply(input));

// Create the model based on the inputs.
const model = tf.model({ inputs: input, outputs: output });

// The model can be used for training, evaluation and prediction.
// For example, the following line runs prediction with the model on
// some fake data.
model.predict(tf.ones([2, 5])).print();

///////////////////////////////////////////////////////////////////////
// SEQUENTIAL MODEL
