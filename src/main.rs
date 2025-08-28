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

fn main() {
    let (w1, w2, b) = update_weights(0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8);
    println!("{w1}, {w2}, {b}");
}
