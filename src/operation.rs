use ndarray::{array, Array2};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::tensor::Tensor;

pub trait Operation: Debug {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>>;
    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>>;
}

pub trait ToArray2 {
    fn to_array2(self) -> Array2<f64>;
}

impl ToArray2 for f64 {
    fn to_array2(self) -> Array2<f64> {
        array![[self]]
    }
}

impl ToArray2 for Vec<f64> {
    fn to_array2(self) -> Array2<f64> {
        Array2::from_shape_vec((self.len(), 1), self).expect("Invalid shape!")
    }
}

impl ToArray2 for Array2<f64> {
    fn to_array2(self) -> Array2<f64> {
        self
    }
}
