use std::collections::HashMap;
use std::collections::hash_map::Keys;
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

    pub fn copy_from(&mut self, other: &Vector<T>) {
        self.map.clear();
        for (t, value) in other.map.iter() {
            self.put(t.clone(), *value);
        }
    }

    pub fn get(&self, t: &T) -> Option<&f64> {
        self.map.get(t)
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

    pub fn iter(&self) -> Keys<T, f64> {
        self.map.keys()
    }

    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for value in self.map.values() {
            sum += value;
        }
        sum
    }

    pub fn get_size(&self) -> usize {
        self.map.len()
    }
}

