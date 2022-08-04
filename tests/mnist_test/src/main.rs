use blowtorch::nn::{loading::NpzWeightLoader, Layer};
use ndarray_npy;
use std::process::ExitCode;
use ordered_float::NotNan;
mod models;

fn main() -> ExitCode {
    let mut loader = NpzWeightLoader::from_path("weights.npz").unwrap();
    let m: models::MnistClassifier<f32> = models::MnistClassifier::new(&mut loader);

    let example_input = ndarray_npy::read_npy("examples/example_7.npy").unwrap();
    // from https://stackoverflow.com/questions/53903318/what-is-the-idiomatic-way-to-get-the-index-of-a-maximum-or-minimum-floating-poin
    let prediction_probabilities: Vec<NotNan<f32>> = m.forward_pass(&example_input).iter()
    .cloned()
    .map(NotNan::new)       // Attempt to convert each f32 to a NotNan
    .filter_map(Result::ok) // Unwrap the `NotNan`s and filter out the `NaN` values 
    .collect();
    let max = prediction_probabilities.iter().max().unwrap();
    let prediction = prediction_probabilities.iter().position(|element| element == max).unwrap();

    let expected_pred = 7;

    if prediction == expected_pred {
        println!("Successfully predicted class: {:?}", prediction);
        ExitCode::SUCCESS
    } else {
        eprintln!("Wrong prediction in rust model: {:?} instead of {:?}", prediction, expected_pred);
        ExitCode::FAILURE
    }
}
