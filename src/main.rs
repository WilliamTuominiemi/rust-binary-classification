fn sigmoid(z: f32) -> f32 {
    return 1.0 / (1.0 + (-z).exp());
}

fn forward_pass(w1: f32, x1: f32, w2: f32, x2: f32, b: f32) -> f32 {
    let z = w1 * x1 + w2 * x2 + b;
    let a = sigmoid(z);
    return a;
}

fn main() {
    let a = forward_pass(1.24, 1.45, 1.11, 1.54, 1.0);
    println!("{a}");
}
