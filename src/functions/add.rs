use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! add {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::Add;
        use crate::operation::Operation;
        use crate::tensor;

        let t1 = tensor!($val1.clone());
        let t2 = tensor!($val2.clone());

        let add = Add::new();
        add.apply(&[t1, t2])
    }};
}

#[derive(Debug, Clone)]
pub struct Add {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Add {
    pub fn new() -> Self {
        Add {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Add {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];
        let b = &inputs[1];
        let add = &a.borrow().arr + &b.borrow().arr;
        let op_name = self.name_manager.clone().borrow_mut().new_name("add");

        tensor!(add, name: &op_name, parents: vec![a.clone(), b.clone()], operation: Box::new(self.clone()))
    }

    fn grad(&self, back_grad: TensorRef, _args: &[TensorRef]) -> Vec<TensorRef> {
        let back_grad_arr = back_grad.borrow().arr.clone();
        let grad = tensor!(back_grad_arr, name: "add_grad");

        vec![grad.clone(), grad]
    }
}
