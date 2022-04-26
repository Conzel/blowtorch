use ndarray_npy::ReadableElement;
use num_traits::{Float, FromPrimitive};
use std::{io::Read, ops::AddAssign};

pub trait FloatLikePrimitive:
    'static + Float + AddAssign + FromPrimitive + ReadableElement
{
}

impl<T> FloatLikePrimitive for T where
    T: 'static + Float + AddAssign + FromPrimitive + ReadableElement
{
}

/// General model trait neural layers
pub trait Layer<I, O> {
    fn forward_pass(&self, input: &I) -> O;
}
