use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! ln {
    ($val1:expr) => {{
        use crate::functions::Ln;
        use crate::operation::Operation;
        use crate::tensor;

        let t = tensor!($val1.clone());

        let ln = Ln::new();
        ln.apply(&[t])
    }};
}

#[derive(Debug, Clone)]
pub struct Ln {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Ln {
    pub fn new() -> Self {
        Ln {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Ln {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let lns = a.borrow().arr.mapv(|v| v.ln());
        let op_name = self.name_manager.clone().borrow_mut().new_name("ln");

        tensor!(lns, name: &op_name, parents: vec![a.clone()], operation: Box::new(self.clone()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let grad_arr = a.borrow().arr.mapv(|x| 1.0 / x);
        let grad = tensor!(back_grad.borrow().arr.clone() * grad_arr, name: "ln_grad");

        vec![grad]
    }
}
