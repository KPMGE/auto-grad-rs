use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! sigmoid {
    ($val1:expr) => {{
        use crate::functions::Sigmoid;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let sigmoid = Sigmoid::new();
        sigmoid.apply(&[t])
    }};
}

#[derive(Debug, Clone)]
pub struct Sigmoid {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Sigmoid {
    fn sigmoid(&self, val: f64) -> f64 {
        1.0 / (1.0 + (-val).exp())
    }
}

impl Operation for Sigmoid {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let sigmoid = a.borrow().arr.mapv(|v| self.sigmoid(v));
        let op_name = self.name_manager.clone().borrow_mut().new_name("sigmoid");

        tensor!(sigmoid, name: &op_name, parents: vec![a.clone()], operation: Box::new(self.clone()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let sigmod_result_arr = a.borrow().arr.mapv(|v| self.sigmoid(v));

        let grad_arr = sigmod_result_arr.clone() * (1.0 - sigmod_result_arr);
        let grad = tensor!(&back_grad.borrow().arr * grad_arr, name: "sigmoid_grad");

        vec![grad]
    }
}
