use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
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
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];
        let b = &inputs[1];

        let mul = a.borrow().arr().dot(b.borrow().arr());

        let op_name = self.name_manager.clone().borrow_mut().new_name("matmul");

        let tensor = TensorBuilder::new(mul.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(MatMul::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0];
        let b = &args[1];

        let a_grad = back_grad.borrow().arr().dot(&b.borrow().arr().t());
        let b_grad = a.borrow().arr().t().dot(back_grad.borrow().arr());

        vec![
            tensor!(a_grad, name: "matmul_grad"),
            tensor!(b_grad, name: "matmul_grad"),
        ]
    }
}
