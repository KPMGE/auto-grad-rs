use std::{fmt::Debug, rc::Rc, vec};

use crate::{functions::Add, operation::Operation, tensor::Tensor, tensor::TensorBuilder};
use std::cell::RefCell;

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let add = Add::new();

    let a = gd_tensor!(vec![1.0, 2.0, 3.0]);
    let b = gd_tensor!(vec![4.0, 5.0, 6.0]);
    let c = gd_tensor!(3.0);

    let d = add.apply(&[a.clone(), b.clone()]);
    let e = add.apply(&[d.clone(), c.clone()]);

    e.borrow_mut().backward(None);

    println!("{:?}", a.borrow().grad());
    println!("{:?}", b.borrow().grad());
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        println!("{}", self.arr());
        Ok(())
    }
}
