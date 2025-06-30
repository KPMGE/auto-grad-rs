use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug)]
pub struct NameManager {
    count: HashMap<String, i32>,
}

impl NameManager {
    pub fn new() -> Self {
        NameManager {
            count: HashMap::new(),
        }
    }

    pub fn new_name(&mut self, name: &str) -> String {
        let n = self.count.entry(name.to_string()).or_insert(0);
        let formatted_name = format!("{}:{}", name, *n);
        *n += 1;
        formatted_name
    }

    pub fn reset(&mut self) {
        self.count.clear();
    }
}

thread_local! {
    pub(crate) static NAME_MANAGER: Rc<RefCell<NameManager>> = Rc::new(RefCell::new(NameManager::new()));
}
