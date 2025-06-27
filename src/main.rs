use std::{f64::consts::PI, fmt::Debug, vec};

use crate::{operation::Operation, tensor::Tensor, tensor::ToTensor};

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let a = gd_tensor!(vec![PI, 0.0, PI / 2.0]);
    let b = sin!(a);
    let c = cos!(a);
    let d = sum!(add!(b, c));

    d.borrow_mut().backward(None);

    println!("{:#?}", a.borrow().grad());
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        println!("{}", self.arr());
        Ok(())
    }
}
