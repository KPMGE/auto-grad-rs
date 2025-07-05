use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! relu {
    ($val1:expr) => {{
        use crate::functions::ReLU;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let relu = ReLU::new();
        relu.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct ReLU {
    name_manager: Rc<RefCell<NameManager>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }

    fn apply(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn grad(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Operation for ReLU {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();

        let relu = a.mapv(|x| ReLU::apply(x));
        let op_name = self.name_manager.clone().borrow_mut().new_name("relu");

        let tensor = TensorBuilder::new(relu)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(ReLU::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0].borrow().arr();
        let relu_grad = a.mapv(|x| ReLU::grad(x)) * back_grad.borrow().arr();
        let grad = tensor!(relu_grad, name: "relu_grad");

        vec![grad]
    }
}
