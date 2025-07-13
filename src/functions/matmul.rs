use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! matrix {
    ( $( $x:expr ),* ) => {{
        use ndarray::array;
        array![$( $x ),*]
    }}
}

#[macro_export]
macro_rules! matmul {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::MatMul;
        use crate::operation::Operation;

        let t1 = tensor!($val1.clone());
        let t2 = tensor!($val2.clone());

        let matmul = MatMul::new();
        matmul.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
pub struct MatMul {
    name_manager: Rc<RefCell<NameManager>>,
}

impl MatMul {
    pub fn new() -> Self {
        MatMul {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for MatMul {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];
        let b = &inputs[1];

        let mul = a.borrow().arr.clone().dot(&b.borrow().arr);
        let op_name = self.name_manager.clone().borrow_mut().new_name("matmul");

        tensor!(mul, name: &op_name, parents: vec![a.clone(), b.clone()], operation: Box::new(MatMul::new()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let b = &args[1];

        let a_grad = back_grad.borrow().arr.clone().dot(&b.borrow().arr.t());
        let b_grad = a.borrow().arr.t().clone().dot(&back_grad.borrow().arr);

        vec![
            tensor!(a_grad, name: "matmul_grad"),
            tensor!(b_grad, name: "matmul_grad"),
        ]
    }
}
