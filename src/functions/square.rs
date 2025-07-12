use std::{cell::RefCell, rc::Rc};

use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor,
    tensor::{TensorBuilder, TensorRef},
};

#[macro_export]
macro_rules! square {
    ($val1:expr) => {{
        use crate::functions::Square;
        use crate::operation::Operation;

        let t = tensor!($val1.clone());

        let square = Square::new();
        square.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Square {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Square {
    pub fn new() -> Self {
        Square {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }
}

impl Operation for Square {
    fn apply(&self, inputs: &[TensorRef]) -> TensorRef {
        let a = &inputs[0];

        let square = a.borrow().arr.mapv(|v| v.powf(2.0));
        let op_name = self.name_manager.clone().borrow_mut().new_name("square");

        let tensor = TensorBuilder::new(square)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Square::new()))
            .build();

        tensor!(tensor)
    }

    fn grad(&self, back_grad: TensorRef, args: &[TensorRef]) -> Vec<TensorRef> {
        let a = &args[0];
        let grad_arr = 2.0 * &a.borrow().arr;
        let grad = tensor!(&back_grad.borrow().arr * grad_arr, name: "square_grad");

        vec![grad]
    }
}
