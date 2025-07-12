use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! prod {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::Prod;
        use crate::operation::Operation;
        use crate::tensor;

        let t1 = tensor!($val1.clone());
        let t2 = tensor!($val2.clone());

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
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Prod {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];
        let b = &inputs[1];
        let product = &a.borrow().arr * &b.borrow().arr;
        let op_name = self.name_manager.clone().borrow_mut().new_name("prod");

        let tensor = TensorBuilder::new(product.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(Prod::new()))
            .build();

        tensor!(tensor)
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let b = &args[1];

        let grad_a = tensor!(&back_grad.borrow().arr * &b.borrow().arr, name: "prod_grad");
        let grad_b = tensor!(&back_grad.borrow().arr * &a.borrow().arr, name: "prod_grad");

        vec![grad_a, grad_b]
    }
}
