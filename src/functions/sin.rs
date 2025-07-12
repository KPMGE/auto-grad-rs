use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! sin {
    ($val1:expr) => {{
        use crate::functions::Sin;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let sin = Sin::new();
        sin.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Sin {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sin {
    pub fn new() -> Self {
        Sin {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Sin {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let sin = a.borrow().arr.sin();
        let op_name = self.name_manager.clone().borrow_mut().new_name("sin");

        let tensor = TensorBuilder::new(sin)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Sin::new()))
            .build();

        tensor!(tensor)
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let grad_arr = &back_grad.borrow().arr * a.borrow().arr.cos();
        let grad = tensor!(grad_arr, name: "sin_grad");

        vec![grad]
    }
}
