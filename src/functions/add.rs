use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! add {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::Add;
        use crate::operation::Operation;

        let t1 = tensor!($val1.clone());
        let t2 = tensor!($val2.clone());

        let add = Add::new();
        add.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
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
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];
        let b = &inputs[1];
        let sum = a.borrow().arr() + b.borrow().arr();
        let op_name = self.name_manager.clone().borrow_mut().new_name("add");

        let tensor = TensorBuilder::new(sum.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(Add::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        _args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let back_grad_arr = back_grad.borrow().arr().clone();
        let grad = tensor!(back_grad_arr, name: "add_grad");

        vec![grad.clone(), grad]
    }
}
