//! Contains the implementation of a simple multi-layer perceptron,
//! also called Linear layer.
//! See <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear>
use ndarray::*;
use num_traits::Float;

/// Rust implementation of a linear layer.

pub struct LinearLayer<F: Float> {
    /// Weight matrix of the kernel
    pub(in crate) weights: Array2<F>,
    pub(in crate) bias: Option<Array1<F>>,
}

impl<F: 'static + Float + std::ops::AddAssign> LinearLayer<F> {
    /// Creates new linear layer.
    /// The weights are given in Pytorch layout.

    pub fn new(
        weights_array: Array2<F>,
        bias_array: Option<Array1<F>>,
    ) -> LinearLayer<F> {
        LinearLayer {
            weights: weights_array,
            bias: bias_array,
        }
    }

    /// Analog to nn.Linear.
    pub fn linear(&self, input_array: &Array2<F>) -> Array2<F> {
        multiply(
            &self.weights,
            self.bias.as_ref(),
            input_array,
        )
    }
}

/// Performs a linear on the given input_array data using this layers parameters.

/// Input:
/// -----------------------------------------------
/// - kernel_weights: weights of shape (F, C)
/// - im2d: Input data of shape (C, D)

/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (F, D)
pub fn multiply<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
) -> Array2<F>
where
    // This trait bound ensures that kernel and im2d can be passed as owned array or view.
    // AsArray just ensures that im2d can be converted to an array view via ".into()".
    // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    V: AsArray<'a, F, Ix2>,
    T: AsArray<'a, F, Ix2>,
{
    // Initialisations
    let im2d_arr: ArrayView2<F> = im2d.into();
    let kernel_weights_arr: ArrayView2<F> = kernel_weights.into();


    let mul = im2d_arr.dot(&kernel_weights_arr.t());
    add_bias(&mul, bias)
}

pub(in crate) fn add_bias<F>(x: &Array2<F>, bias: Option<&Array1<F>>) -> Array2<F>
where
    F: 'static + Float + std::ops::AddAssign,
{
    if let Some(bias_array) = bias {
        assert!(
            bias_array.shape()[0] == x.shape()[1],
            "Bias array has the wrong shape {:?} for vec of shape {:?}",
            bias_array.shape(),
            x.shape()
        );
        // Yes this is really necessary. Broadcasting with ndarray-rust
        // starts at the right side of the shape, so we have to add
        // the axes by hand (else it thinks that it should compare the
        // output width and the bias channels).
        (x + &bias_array
            .clone()
            .insert_axis(Axis(0))
            .broadcast(x.shape())
            .unwrap())
            .into_dimensionality()
            .unwrap()
    } else {
        x.clone()
    }
}