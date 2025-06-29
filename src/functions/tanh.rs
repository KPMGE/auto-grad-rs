use std::{cell::RefCell, rc::Rc};

use ndarray::Array2;

use crate::{
    gd_tensor,
    name_manager::NameManager,
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! tanh {
    ($val1:expr) => {{
        use crate::functions::Tanh;

        let t = gd_tensor!($val1.clone());

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
            name_manager: Rc::new(RefCell::new(NameManager::new())),
        }
    }
}

impl Operation for Tanh {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();

        let tanh = a.mapv(f64::tanh);
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
        let a = &args[0].borrow().arr();
        let tanh_squared = a.mapv(|v| v.tanh() * v.tanh());

        let grad_arr = 1.0 - tanh_squared;
        let grad = gd_tensor!(back_grad.borrow().arr() * grad_arr, name: "tanh_grad");

        vec![grad]
    }
}
