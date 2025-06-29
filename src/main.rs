use std::vec;

use crate::{operation::Operation, tensor::Tensor};

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let v = gd_tensor!(vec![1.0, 2.0, 3.0]);
    let w = exp!(v);

    println!("{:#?}", w);

    w.borrow_mut().backward(None);

    println!("{:#?}", v.borrow().grad());

    println!("=========================================");

    let v2 = gd_tensor!(vec![3.0, 1.0, 0.0, 2.0]);
    let w2 = square!(v2);

    println!("{:#?}", w2);

    w2.borrow_mut().backward(None);

    println!("{:#?}", v2.borrow().grad());
}
