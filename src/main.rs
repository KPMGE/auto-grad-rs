use std::{cell::RefCell, rc::Rc, vec};

use crate::{functions::Add, operation::Operation, tensor::TensorBuilder};

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let add = Add::new();

    let a_arr = vec![1.0, 2.0, 3.0];
    let arr_b = vec![4.0, 5.0, 6.0];

    let a = Rc::new(RefCell::new(
        TensorBuilder::new(a_arr.clone()).name("a").build(),
    ));
    let b = Rc::new(RefCell::new(
        TensorBuilder::new(arr_b.clone()).name("b").build(),
    ));

    let c = Rc::new(RefCell::new(TensorBuilder::new(3.0).name("c").build()));

    let d = add.apply(&[a.clone(), b.clone()]);

    let e = add.apply(&[d.clone(), c.clone()]);

    e.borrow_mut().backward(None);

    println!("{:?}", a.borrow().grad());
    println!("{:?}", b.borrow().grad());
}
