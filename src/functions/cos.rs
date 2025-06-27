use std::{cell::RefCell, rc::Rc};

use crate::{
    gd_tensor,
    name_manager::NameManager,
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! cos {
    ($val1:expr) => {{
        use crate::functions::Cos;

        let t = gd_tensor!($val1.clone());

        let cos = Cos::new();
        cos.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Cos {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Cos {
    pub fn new() -> Self {
        Cos {
            name_manager: Rc::new(RefCell::new(NameManager::new())),
        }
    }
}

impl Operation for Cos {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();

        let cos_arr = a.cos();
        let op_name = self.name_manager.clone().borrow_mut().new_name("cos");

        let tensor = TensorBuilder::new(cos_arr)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Cos::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0].borrow().arr();
        let grad_arr = back_grad.borrow().arr() * -a.sin();
        let grad = gd_tensor!(grad_arr, name: "cos_grad");

        vec![grad]
    }
}
