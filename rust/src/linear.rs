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

fn arr_allclose<D: Dimension>(current: &Array<f32, D>, target: &Array<f32, D>) -> bool {
    assert_eq!(
        current.shape(),
        target.shape(),
        "\ngiven array had shape {:?}, but target had shape {:?}",
        current.shape(),
        target.shape()
    );
    (current - target).map(|x| (*x as f32).abs()).sum() < 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let test_img = array![[-1.0643, -0.8746, -0.5266,  0.6039], 
        [ 0.7219, -0.8092,  0.1590, -0.2309], 
        [ 0.6337, -1.4233,  0.7101, -0.9875]];

        let kernel: Array2<f32> = Array::from_shape_vec(
            (2, 4), 
            vec![0.0494,  0.1638, -0.3392, -0.3255, -0.1564,  0.3991,  0.1288,  0.0259],).unwrap();
        let linear_layer = LinearLayer::new(kernel, None);
        let linear_output = linear_layer.linear(&test_img);
        let output: Array2<f32> = Array::from_shape_vec((3,2), vec![-0.21378263, -0.23478141, -0.07565995, -0.42135799, -0.12126643, -0.60126508]).unwrap();
        assert!(
            arr_allclose(&linear_output, &output),
            "{:?} was not equal to {:?}",
            linear_output,
            output
        );
    }

    #[test]
    fn test_linear1() {
        let test_img = array![[-1.0643, -0.8746, -0.5266,  0.6039], 
        [ 0.7219, -0.8092,  0.1590, -0.2309], 
        [ 0.6337, -1.4233,  0.7101, -0.9875]];

        let kernel: Array2<f32> = Array::from_shape_vec(
            (2, 4), 
            vec![0.0494,  0.1638, -0.3392, -0.3255, -0.1564,  0.3991,  0.1288,  0.0259],).unwrap();
        let linear_layer = LinearLayer::new(kernel, Some(Array::from_shape_vec((2,), vec![ 0.3759, -0.0778],).unwrap()));
        let linear_output = linear_layer.linear(&test_img);
        let output: Array2<f32> = Array::from_shape_vec((3,2), vec![ 0.1621, -0.3126,  0.3003, -0.4992, 0.2547, -0.6791]).unwrap();
        assert!(
            arr_allclose(&linear_output, &output),
            "{:?} was not equal to {:?}",
            linear_output,
            output
        );
    }
}