use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! exp {
    ($val1:expr) => {{
        use crate::functions::Exp;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let exp = Exp::new();
        exp.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Exp {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Exp {
    pub fn new() -> Self {
        Exp {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Exp {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let exp = a.borrow().arr.mapv(|v| v.exp());
        let op_name = self.name_manager.clone().borrow_mut().new_name("exp");

        tensor!(exp, name: &op_name, parents: vec![a.clone()], operation: Box::new(Exp::new()))
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let grad_arr = a.borrow().arr.exp();
        let grad = tensor!(back_grad.borrow().arr.clone() * grad_arr, name: "exp_grad");

        vec![grad]
    }
}
