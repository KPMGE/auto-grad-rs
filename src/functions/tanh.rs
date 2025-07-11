use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
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
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];

        let tanh = a.borrow().arr().mapv(f64::tanh);
        let op_name = self.name_manager.clone().borrow_mut().new_name("tanh");

        let tensor = TensorBuilder::new(tanh)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Tanh::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0];
        let tanh_squared = a.borrow().arr().mapv(|v| v.tanh() * v.tanh());

        let grad_arr = 1.0 - tanh_squared;
        let grad = tensor!(back_grad.borrow().arr() * grad_arr, name: "tanh_grad");

        vec![grad]
    }
}
