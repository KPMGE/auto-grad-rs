use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! relu {
    ($val1:expr) => {{
        use crate::functions::ReLU;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let relu = ReLU::new();
        relu.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct ReLU {
    name_manager: Rc<RefCell<NameManager>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }

    fn apply(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn grad(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Operation for ReLU {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let relu = a.borrow().arr.mapv(|x| ReLU::apply(x));
        let op_name = self.name_manager.clone().borrow_mut().new_name("relu");

        tensor!(relu, name: &op_name, parents: vec![a.clone()], operation: Box::new(ReLU::new()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let relu_grad = a.borrow().arr.mapv(|x| ReLU::grad(x)) * &back_grad.borrow().arr;
        let grad = tensor!(relu_grad, name: "relu_grad");

        vec![grad]
    }
}
