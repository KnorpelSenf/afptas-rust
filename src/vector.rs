use std::collections::HashMap;
use std::hash::Hash;

pub struct Vector<T> {
    pub map: HashMap<T, f64>,
}

impl<T: Eq + PartialEq + Hash + Clone> Vector<T> {
    pub fn new() -> Self {
        Vector {
            map: HashMap::new(),
        }
    }

    pub fn add(&self, other: &Vector<T>) -> Vector<T> {
        let mut res = Vector::new();

        for (t, value) in self.map.iter() {
            if let Some(other_value) = other.map.get(t) {
                res.put(t.clone(), value + other_value);
            } else {
                res.put(t.clone(), *value);
            }
        }

        res
    }

    pub fn scale(&self, tau: f64) -> Vector<T> {
        let mut res = Vector::new();
        for (t, value) in self.map.iter() {
            res.put(t.clone(), value * tau);
        }
        res
    }

    pub fn put(&mut self, t: T, n: f64) {
        if n == 0.0 {
            self.map.remove(&t);
        } else {
            self.map.insert(t, n);
        }
    }
}
