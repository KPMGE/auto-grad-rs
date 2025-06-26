use std::{fmt::Debug, vec};

use crate::{
    operation::Operation, tensor::Tensor,
};

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let a = gd_tensor!(vec![1.0, 2.0, 3.0]);
    let b = gd_tensor!(vec![4.0, 5.0, 6.0]);
    let c = sub!(a, b);
    let d = sub!(c, 3.0);

    d.borrow_mut().backward(None);

    println!("{:?}", a.clone().borrow().grad());
    println!("{:?}", b.clone().borrow().grad());
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        println!("{}", self.arr());
        Ok(())
    }
}
