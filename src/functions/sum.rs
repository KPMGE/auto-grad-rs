use std::{cell::RefCell, rc::Rc};

use ndarray::Array2;

use crate::tensor;
use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! sum {
    ($val1:expr) => {{
        use crate::functions::Sum;
        use crate::operation::Operation;
        use crate::tensor;

        let t = tensor!($val1.clone());

        let sum = Sum::new();
        sum.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Sum {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Sum {
    pub fn new() -> Self {
        Sum {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Sum {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];

        let sum = a.borrow().arr().sum();
        let op_name = self.name_manager.clone().borrow_mut().new_name("sum");

        let tensor = TensorBuilder::new(sum)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Sum::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let input_dim = args[0].borrow().arr().raw_dim();
        let grad_arr = Array2::from_elem(input_dim, 1.0);
        let grad = tensor!(grad_arr, name: "sum_grad");

        vec![grad]
    }
}
