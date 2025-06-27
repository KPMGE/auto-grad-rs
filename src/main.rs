use std::vec;

use crate::operation::Operation;

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let v = gd_tensor!(vec![-1.0, 0.0, 1.0, 3.0]);
    let w = tanh!(v);

    println!("{:#?}", w);

    w.borrow_mut().backward(None);

    println!("{:#?}", v.borrow().grad());
}
