fn sigmoid(z: f32) -> f32 {
    return 1.0 / (1.0 + (-z).exp());
}

fn main() {
    let s = sigmoid(1.24);
    println!("{s}");
}