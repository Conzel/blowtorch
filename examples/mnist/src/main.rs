use blowtorch::ndarray::*;
use blowtorch::nn::{loading::NpzWeightLoader, Layer};
use ndarray_npy;
mod models;

fn main() {
    let mut loader = NpzWeightLoader::from_path("weights.npz").unwrap();
    let m: models::MnistClassifier<f32> = models::MnistClassifier::new(&mut loader);

    let example_input = ndarray_npy::read_npy("examples/example_4.npy").unwrap();
    let predictions = m.forward_pass(&example_input);

    let pred_array = predictions.into_shape((1, 10)).unwrap();
    for (i, row) in pred_array.axis_iter(Axis(0)).enumerate() {
        let (max_idx, max_val) =
            row.iter()
                .enumerate()
                .fold((0, row[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        println!("Predicted Class: {:?}", max_idx)
    }
}
