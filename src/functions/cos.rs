use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
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

#[derive(Debug)]
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
        let grad = tensor!(grad_arr, name: "cos_grad");

        vec![grad]
    }
}
