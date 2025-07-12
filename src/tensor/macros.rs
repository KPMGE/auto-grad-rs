use crate::{
    operation::ToArray2,
    tensor::{Tensor, TensorBuilder, TensorRef},
};

impl ToTensor for Tensor {
    fn to_tensor(self) -> TensorRef {
        TensorRef::new(self)
    }
}

pub trait ToTensor {
    fn to_tensor(self) -> TensorRef;
}

impl ToTensor for TensorRef {
    fn to_tensor(self) -> TensorRef {
        self
    }
}

impl<'a> ToTensor for &'a TensorRef {
    fn to_tensor(self) -> TensorRef {
        self.clone()
    }
}

impl<T: ToArray2> ToTensor for T {
    fn to_tensor(self) -> TensorRef {
        let tensor = TensorBuilder::new(self.to_array2()).build();
        TensorRef::new(tensor)
    }
}

#[macro_export]
macro_rules! tensor {
    ($val:expr) => {{
        use $crate::tensor::ToTensor;

        $val.to_tensor()
    }};

    ($val:expr $(, name: $name:expr)? $(, requires_grad: $grad:expr)? $(, parents: $parents:expr)? ) => {{
        use $crate::tensor::TensorRef;

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
        TensorRef::new(builder.build())
    }};
}
