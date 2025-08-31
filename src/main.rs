use rand::Rng;

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn forward_pass(w1: f32, x1: f32, w2: f32, x2: f32, b: f32) -> f32 {
    let z = w1 * x1 + w2 * x2 + b;
    let a = sigmoid(z);
    a
}

fn compute_loss(y_true: f32, y_pred: f32) -> f32 {
    let epsilon = f32::powf(10.0, -12.0);
    let a = y_pred.clamp(epsilon, 1.0 - epsilon);
    let loss = -(y_true * a.ln() + (1.0 - y_true) * (1.0 - a).ln());
    loss
}

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
    let error = y_pred - y_true;
    let dw1 = error * x1;
    let dw2 = error * x2;
    let db = error;

    let updated_w1 = w1 - learning_rate * dw1;
    let updated_w2 = w2 - learning_rate * dw2;
    let updated_b = b - learning_rate * db;

    (updated_w1, updated_w2, updated_b)
}

fn initialize_weights() -> (f32, f32, f32) {
    let mut rng = rand::rng();
    let w1: f32 = rng.random_range(-0.5..0.5);
    let w2: f32 = rng.random_range(-0.5..0.5);
    let b: f32 = rng.random_range(-0.5..0.5);

    (w1, w2, b)
}

fn train(x: Vec<(f32, f32)>, y: Vec<f32>, epochs: i32, learning_rate: f32) -> (f32, f32, f32) {
    let mut w1;
    let mut w2;
    let mut b;

    (w1, w2, b) = initialize_weights();

    let zipped: Vec<((f32, f32), f32)> = x.iter().cloned().zip(y.iter().cloned()).collect();

    for epoch in 0..epochs {
        println!("epoch: {}", epoch + 1);

        for ((x1, x2), y_true) in &zipped {
            let y_pred = forward_pass(w1, *x1, w2, *x2, b);
            let (new_w1, new_w2, new_b) =
                update_weights(w1, *x1, b, w2, *x2, *y_true, y_pred, learning_rate);
            w1 = new_w1;
            w2 = new_w2;
            b = new_b;
        }
    }

    (w1, w2, b)
}

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
