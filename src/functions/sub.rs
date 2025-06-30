use std::{cell::RefCell, rc::Rc};

use crate::{
    gd_tensor,
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! sub {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::Sub;

        let t1 = gd_tensor!($val1.clone());
        let t2 = gd_tensor!($val2.clone());

        let sub = Sub::new();
        sub.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
pub struct Sub {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sub {
    pub fn new() -> Self {
        Sub {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Sub {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();
        let b = &inputs[1].borrow().arr();
        let sub = a - b;
        let op_name = self.name_manager.clone().borrow_mut().new_name("sub");

        let tensor = TensorBuilder::new(sub.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(Sub::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        _args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let back_grad_arr = back_grad.borrow().arr();

        let grad_a = gd_tensor!(back_grad_arr.clone(), name: "sub_grad");
        let grad_b = gd_tensor!(&back_grad_arr * -1.0, name: "sub_grad");

        vec![grad_a, grad_b]
    }
}
