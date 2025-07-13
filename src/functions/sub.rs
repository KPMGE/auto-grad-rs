use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! sub {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::Sub;
        use crate::operation::Operation;

        let t1 = tensor!($val1.clone());
        let t2 = tensor!($val2.clone());

        let sub = Sub::new();
        sub.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
pub struct Sub {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sub {
    pub fn new() -> Self {
        Sub {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Sub {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];
        let b = &inputs[1];

        let sub = &a.borrow().arr - &b.borrow().arr;
        let op_name = self.name_manager.clone().borrow_mut().new_name("sub");

        tensor!(sub, name: &op_name, parents: vec![a.clone(), b.clone()], operation: Box::new(Sub::new()))
    }

    fn grad(&self, back_grad: TensorRef, _args: &[TensorRef]) -> Vec<TensorRef> {
        let grad_a = tensor!(back_grad.borrow().arr.clone(), name: "sub_grad");
        let grad_b = tensor!(&back_grad.borrow().arr * -1.0, name: "sub_grad");

        vec![grad_a, grad_b]
    }
}
