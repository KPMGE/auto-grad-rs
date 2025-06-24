use std::{cell::RefCell, collections::HashMap};

#[derive(Debug)]
pub struct NameManager {
    count: RefCell<HashMap<String, i32>>,
}

impl NameManager {
    pub fn new() -> Self {
        NameManager {
            count: RefCell::new(HashMap::new()),
        }
    }

    pub fn new_name(&self, name: &str) -> String {
        let map = self.count.borrow();
        let n = map.get(name).unwrap_or(&0);
        self.count.borrow_mut().insert(name.to_string(), *n);
        format!("{name}:{n}")
    }

    pub fn reset(&self) {
        self.count.borrow_mut().clear();
    }
}
