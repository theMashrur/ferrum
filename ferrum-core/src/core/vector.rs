use std::fmt;
use std::ops;

#[derive(Debug, Clone)]
pub struct Vector<T> {
    pub data: Vec<T>,
    pub size: usize,
}

pub type RealVector = Vector<f64>;

impl<T> fmt::Display for Vector<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const EDGE_ITEMS: usize = 2;

        write!(f, "[")?;

        if self.data.len() <= EDGE_ITEMS * 2 {
            for (i, value) in self.data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", value)?;
            }
        } else {
            for (i, value) in self.data.iter().take(EDGE_ITEMS).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", value)?;
            }

            write!(f, ", ...")?;

            for value in self.data.iter().skip(self.data.len() - EDGE_ITEMS) {
                write!(f, ", {}", value)?;
            }
        }

        write!(f, "]")
    }
}

pub trait VectorRead<T> {
    fn size(&self) -> usize;
    fn get(&self, index: usize) -> &T;
}

impl<T> VectorRead<T> for Vector<T> {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> &T {
        assert!(index < self.size, "Index out of bounds");
        &self.data[index]
    }
}

impl<T> Vector<T> {
    pub fn new(size: usize) -> Self {
        let data = Vec::with_capacity(size);
        Vector { data, size }
    }

    pub fn dot(&self, other: &Vector<T>) -> T
    where
        T: ops::Mul<Output = T> + ops::Add<Output = T> + Copy,
    {
        assert_eq!(
            self.size(),
            other.size(),
            "Vectors must be of the same size"
        );
        let mut result = self.data[0] * other.data[0];
        for i in 1..self.size() {
            result = result + (self.data[i] * other.data[i]);
        }
        result
    }
}

// --------------- Arithmetic Operations ----------------

pub fn add_vectors_core<T>(lhs: &Vector<T>, rhs: &Vector<T>, out: &mut Vector<T>)
where
    T: ops::Add<Output = T> + Copy,
{
    assert_eq!(lhs.size(), rhs.size(), "Vectors must be of the same size");
    assert_eq!(
        lhs.size(),
        out.size(),
        "Output vector must be of the same size as input vectors"
    );

    out.data.clear();
    out.data.reserve(lhs.size());

    for i in 0..lhs.size() {
        out.data.push(lhs.data[i] + rhs.data[i]);
    }
}

impl<'a, 'b, T> ops::Add<&'b Vector<T>> for &'a Vector<T>
where
    T: ops::Add<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn add(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert_eq!(self.size(), rhs.size(), "Vectors must be of the same size");
        let mut result = Vector::new(self.size());
        add_vectors_core(self, rhs, &mut result);
        result
    }
}

pub fn sub_vectors_core<T>(lhs: &Vector<T>, rhs: &Vector<T>, out: &mut Vector<T>)
where
    T: ops::Sub<Output = T> + Copy,
{
    assert_eq!(lhs.size(), rhs.size());
    assert_eq!(
        lhs.size(),
        out.size(),
        "Output vector must be of the same size as input vectors"
    );

    out.data.clear();
    out.data.reserve(lhs.size());

    for i in 0..lhs.size() {
        out.data.push(lhs.data[i] - rhs.data[i]);
    }
}

impl<'a, 'b, T> ops::Sub<&'b Vector<T>> for &'a Vector<T>
where
    T: ops::Sub<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn sub(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert_eq!(self.size(), rhs.size(), "Vectors must be of the same size");
        let mut result = Vector::new(self.size());
        sub_vectors_core(self, rhs, &mut result);
        result
    }
}

pub fn scalar_mul_vectors_core<T>(vector: &Vector<T>, scalar: T, out: &mut Vector<T>)
where
    T: ops::Mul<Output = T> + Copy,
{
    out.data.clear();
    out.data.reserve(vector.size());

    for i in 0..vector.size() {
        out.data.push(vector.data[i] * scalar);
    }
}

impl<'a, T> ops::Mul<T> for &'a Vector<T>
where
    T: ops::Mul<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn mul(self, scalar: T) -> Vector<T> {
        let mut result = Vector::new(self.size());
        scalar_mul_vectors_core(self, scalar, &mut result);
        result
    }
}

// --------------- Unit Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let v1 = Vector {
            data: vec![1, 2, 3],
            size: 3,
        };
        let v2 = Vector {
            data: vec![4, 5, 6],
            size: 3,
        };
        let result = &v1 + &v2;
        assert_eq!(result.data, vec![5, 7, 9]);
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector {
            data: vec![5, 7, 9],
            size: 3,
        };
        let v2 = Vector {
            data: vec![1, 2, 3],
            size: 3,
        };
        let result = &v1 - &v2;
        assert_eq!(result.data, vec![4, 5, 6]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let v = Vector {
            data: vec![1, 2, 3],
            size: 3,
        };
        let scalar = 2;
        let result = &v * scalar;
        assert_eq!(result.data, vec![2, 4, 6]);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector {
            data: vec![1, 2, 3],
            size: 3,
        };
        let v2 = Vector {
            data: vec![4, 5, 6],
            size: 3,
        };
        let result = v1.dot(&v2);
        assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 32
    }
}
