use std::{cell::RefCell, rc::Rc};

use ndarray::Array2;

use crate::{
    tensor,
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! sigmoid {
    ($val1:expr) => {{
        use crate::functions::Sigmoid;

        let t = tensor!($val1.clone());

        let sigmoid = Sigmoid::new();
        sigmoid.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Sigmoid {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Sigmoid {
    fn sigmoid(&self, val: f64) -> f64 {
        1.0 / (1.0 + (-val).exp())
    }
}

impl Operation for Sigmoid {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();

        let sigmoid = a.mapv(|v| self.sigmoid(v));
        let op_name = self.name_manager.clone().borrow_mut().new_name("sigmoid");

        let tensor = TensorBuilder::new(sigmoid)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Sigmoid::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0].borrow().arr();
        let sigmod_result_arr = a.mapv(|v| self.sigmoid(v));

        let grad_arr = sigmod_result_arr.clone() * (1.0 - sigmod_result_arr);
        let grad = tensor!(back_grad.borrow().arr() * grad_arr, name: "sigmoid_grad");

        vec![grad]
    }
}
