use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! tanh {
    ($val1:expr) => {{
        use crate::functions::Tanh;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let tanh = Tanh::new();
        tanh.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Tanh {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Tanh {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let tanh = a.borrow().arr.mapv(f64::tanh);
        let op_name = self.name_manager.clone().borrow_mut().new_name("tanh");

        tensor!(tanh, name: &op_name, parents: vec![a.clone()], operation: Box::new(Tanh::new()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let tanh_squared = a.borrow().arr.mapv(|v| v.tanh() * v.tanh());

        let grad_arr = 1.0 - tanh_squared;
        let grad = tensor!(&back_grad.borrow().arr * grad_arr, name: "tanh_grad");

        vec![grad]
    }
}
