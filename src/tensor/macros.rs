use crate::tensor::TensorBuilder;
use std::{cell::RefCell, rc::Rc};

#[macro_export]
macro_rules! gd_tensor {
    ($val:expr $(, name: $name:expr)? $(, requires_grad: $grad:expr)? $(, parents: $parents:expr)? ) => {{
        let builder = TensorBuilder::new($val);
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
