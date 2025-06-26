use std::{cell::RefCell, rc::Rc};

use crate::{
    gd_tensor, name_manager::NameManager, operation::Operation, tensor::{Tensor, TensorBuilder}
};

#[macro_export]
macro_rules! prod {
    ($val1:expr, $val2:expr) => {{
      use crate::functions::Prod;

        let t1 = gd_tensor!($val1.clone());
        let t2 = gd_tensor!($val2.clone());

        let prod = Prod::new();
        prod.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
pub struct Prod {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Prod {
    pub fn new() -> Self {
        Prod {
            name_manager: Rc::new(RefCell::new(NameManager::new())),
        }
    }
}

impl Operation for Prod {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();
        let b = &inputs[1].borrow().arr();
        let product = a * b;
        let op_name = self.name_manager.clone().borrow_mut().new_name("prod");

        let tensor = TensorBuilder::new(product.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(Prod::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = args[0].borrow().arr();
        let b = args[1].borrow().arr();
        let back_grad_arr = back_grad.borrow().arr();

        let grad_a = gd_tensor!(back_grad_arr.clone() * b, name: "prod_grad");
        let grad_b = gd_tensor!(back_grad_arr * a, name: "prod_grad");


        vec![grad_a, grad_b]
    }
}
