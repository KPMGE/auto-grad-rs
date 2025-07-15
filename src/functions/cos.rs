use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! cos {
    ($val1:expr) => {{
        use crate::functions::Cos;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let cos = Cos::new();
        cos.apply(&[t])
    }};
}

#[derive(Debug, Clone)]
pub struct Cos {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Cos {
    pub fn new() -> Self {
        Cos {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Cos {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];
        let cos_arr = a.borrow().arr.cos();
        let op_name = self.name_manager.clone().borrow_mut().new_name("cos");

        tensor!(cos_arr, name: &op_name, parents: vec![a.clone()], operation: Box::new(self.clone()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let grad_arr = back_grad.borrow().arr.clone() * -a.borrow().arr.sin();
        let grad = tensor!(grad_arr, name: "cos_grad");

        vec![grad]
    }
}
