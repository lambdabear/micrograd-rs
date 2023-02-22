use micrograd::{engine::Scalar, nn::MLP};

fn main() {
    let mut rng = rand::thread_rng();
    let mut mlp = MLP::new(3, &[4, 4, 1], &mut rng);

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];

    for _ in 0..1000 {
        let xs = xs.map(|x| x.map(|d| Scalar::new(d, "")));
        let ys = [1.0, -1.0, -1.0, 1.0].map(|d| Scalar::new(d, ""));
        let ypred = xs.map(|x| mlp.output(x.to_vec()).unwrap());
        let loss = ypred
            .iter()
            .enumerate()
            .map(|(i, yp)| (yp[0].clone() - ys[i].clone()).powi(2))
            .fold(Scalar::new(0.0, ""), |acc, s| acc + s);

        println!(
            "Predict: {:?}\nLoss: {:?}",
            ypred.map(|scalars| { scalars.iter().map(|s| { s.data() }).collect::<Vec<f32>>()[0] }),
            loss.data(),
        );

        loss.backward();

        for p in mlp.parameters() {
            let data = p.data();

            p.set_data(data + (-0.001) * p.grad())
        }
    }

    let parameters: Vec<f32> = mlp.parameters().iter().map(|s| s.data()).collect();
    println!("Parameters: {parameters:?}")
}
