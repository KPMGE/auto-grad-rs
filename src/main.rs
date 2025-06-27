use std::{fmt::Debug, vec};

use crate::{operation::Operation, tensor::Tensor};

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let a = gd_tensor!(vec![3.0, 1.0, 0.0, 2.0]);
    let b = add!(prod!(a, 3.0), a);
    let c = sum!(b);

    c.borrow_mut().backward(None);

    println!("{:?}", a.clone().borrow().grad());
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        println!("{}", self.arr());
        Ok(())
    }
}
