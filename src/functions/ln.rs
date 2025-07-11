use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
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

#[derive(Debug)]
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
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];

        let lns = a.borrow().arr().mapv(|v| v.ln());
        let op_name = self.name_manager.clone().borrow_mut().new_name("ln");

        let tensor = TensorBuilder::new(lns)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Ln::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0];
        let grad_arr = a.borrow().arr().mapv(|x| 1.0 / x);
        let grad = tensor!(back_grad.borrow().arr() * grad_arr, name: "ln_grad");

        vec![grad]
    }
}
