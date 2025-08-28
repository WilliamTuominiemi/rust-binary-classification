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

fn main() {
    let l = compute_loss(0.6, 0.8);
    println!("{l}");
}
