use crate::{operation::ToArray2, tensor::{Tensor, TensorBuilder}};
use std::{cell::RefCell, rc::Rc};

pub trait ToTensor {
    fn to_tensor(self) -> Rc<RefCell<Tensor>>;
}

impl ToTensor for Rc<RefCell<Tensor>> {
    fn to_tensor(self) -> Rc<RefCell<Tensor>> {
        self.clone()
    }
}

impl<'a> ToTensor for &'a Rc<RefCell<Tensor>> {
    fn to_tensor(self) -> Rc<RefCell<Tensor>> {
        self.clone()
    }
}

impl<T: ToArray2> ToTensor for T {
    fn to_tensor(self) -> Rc<RefCell<Tensor>> {
        let tensor = TensorBuilder::new(self.to_array2()).build();
        Rc::new(RefCell::new(tensor))
    }
}

#[macro_export]
macro_rules! gd_tensor {
    ($val:expr) => {{
        use $crate::tensor::ToTensor;
        $val.to_tensor()
    }};

    ($val:expr $(, name: $name:expr)? $(, requires_grad: $grad:expr)? $(, parents: $parents:expr)? ) => {{
        let mut builder = TensorBuilder::new($val);
        $(
            builder = builder.name($name);
        )?
        $(
            builder = builder.requires_grad($grad);
        )?
        $(
            builder = builder.parents($parents);
        )?
        Rc::new(RefCell::new(builder.build()))
    }};
}