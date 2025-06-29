use std::{cell::RefCell, rc::Rc};

use crate::{
    gd_tensor,
    name_manager::NameManager,
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! matrix {
    ( $( $x:expr ),* ) => {{
        use ndarray::array;
        array![$( $x ),*]
    }}
}

#[macro_export]
macro_rules! matmul {
    ($val1:expr, $val2:expr) => {{
        use crate::functions::MatMul;
        use crate::operation::Operation;

        let t1 = gd_tensor!($val1.clone());
        let t2 = gd_tensor!($val2.clone());

        let matmul = MatMul::new();
        matmul.apply(&[t1, t2])
    }};
}

#[derive(Debug)]
pub struct MatMul {
    name_manager: Rc<RefCell<NameManager>>,
}

impl MatMul {
    pub fn new() -> Self {
        MatMul {
            name_manager: Rc::new(RefCell::new(NameManager::new())),
        }
    }
}

impl Operation for MatMul {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0].borrow().arr();
        let b = &inputs[1].borrow().arr();

        let mul = a.dot(b);

        let op_name = self.name_manager.clone().borrow_mut().new_name("matmul");

        let tensor = TensorBuilder::new(mul.clone())
            .name(&op_name)
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(MatMul::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let a = &args[0].borrow().arr();
        let b = &args[1].borrow().arr();
        let back_grad_arr = back_grad.borrow().arr();

        let a_grad = back_grad_arr.dot(&b.t());
        let b_grad = a.t().dot(&back_grad_arr);

        vec![gd_tensor!(a_grad), gd_tensor!(b_grad)]
    }
}
