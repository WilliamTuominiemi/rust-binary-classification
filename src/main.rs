use rand::Rng;

// Applies the sigmoid activation function, which maps any real number into the range (0, 1).
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

// Performs a forward pass of a neuron. It computes a weighted sum of the inputs,
// adds the bias, and passes the result through the sigmoid activation function.
fn forward_pass(w1: f32, x1: f32, w2: f32, x2: f32, b: f32) -> f32 {
    let z: f32 = w1 * x1 + w2 * x2 + b;
    let a: f32 = sigmoid(z);
    a
}

// Updates the weights and bias using gradient descent. It calculates
// the prediction error and adjusts the weights and bias in the opposite
// direction of the gradient to reduce the error.
fn update_weights(
    w1: f32,
    x1: f32,
    b: f32,
    w2: f32,
    x2: f32,
    y_true: f32,
    y_pred: f32,
    learning_rate: f32,
) -> (f32, f32, f32) {
    let error: f32 = y_pred - y_true;
    let dw1: f32 = error * x1;
    let dw2: f32 = error * x2;
    let db: f32 = error;

    let updated_w1: f32 = w1 - learning_rate * dw1;
    let updated_w2: f32 = w2 - learning_rate * dw2;
    let updated_b: f32 = b - learning_rate * db;

    (updated_w1, updated_w2, updated_b)
}

// Ensures that the perceptron starts with non-zero parameters.
fn initialize_weights() -> (f32, f32, f32) {
    let mut rng = rand::rng();
    let w1: f32 = rng.random_range(-0.5..0.5);
    let w2: f32 = rng.random_range(-0.5..0.5);
    let b: f32 = rng.random_range(-0.5..0.5);

    (w1, w2, b)
}

// The training process consists of multiple epochs where each input sample
// is fed forward, and the weights/bias are updated using gradient descent.
fn train(x: Vec<(f32, f32)>, y: Vec<f32>, epochs: i32, learning_rate: f32) -> (f32, f32, f32) {
    let mut w1: f32;
    let mut w2: f32;
    let mut b: f32;

    (w1, w2, b) = initialize_weights();

    let zipped: Vec<((f32, f32), f32)> = x.iter().cloned().zip(y.iter().cloned()).collect();

    for epoch in 0..epochs {
        println!("epoch: {}", epoch + 1);

        for ((x1, x2), y_true) in &zipped {
            let y_pred: f32 = forward_pass(w1, *x1, w2, *x2, b);
            let (new_w1, new_w2, new_b) =
                update_weights(w1, *x1, b, w2, *x2, *y_true, y_pred, learning_rate);
            w1 = new_w1;
            w2 = new_w2;
            b = new_b;
        }
    }

    (w1, w2, b)
}

// Makes predictions using the trained perceptron and compares them to the true.
fn predict(x: Vec<(f32, f32)>, y: Vec<f32>, w1: f32, w2: f32, b: f32) {
    for (index, (x1, x2)) in x.iter().enumerate() {
        let prediction: f32 = forward_pass(w1, *x1, w2, *x2, b);
        println!("Prediction: {} | True: {}", prediction, y[index]);
    }
    return;
}

fn main() {
    let x: Vec<(f32, f32)> = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
    let y: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0];

    let (w1, w2, b) = train(x.clone(), y.clone(), 500, 0.25);

    predict(x.clone(), y.clone(), w1, w2, b);
}
