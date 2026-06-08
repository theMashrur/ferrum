use std::fmt;
use std::ops;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64
}


impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{0}i + {1}j", self.real, self.imag)
    }
}


impl ops::Add<Complex> for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl ops::Sub<Complex> for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Complex {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl ops::Add<f64> for Complex {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Complex {
            real: self.real + rhs,
            imag: self.imag
        }
    }
}

impl ops::Sub<f64> for Complex {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Complex {
            real: self.real - rhs,
            imag: self.imag
        }
    }
}

impl ops::Mul<Complex> for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Complex {
            real: self.real * rhs,
            imag: self.imag * rhs
        }
    }
}

impl From<f64> for Complex {
    fn from(value: f64) -> Self {
        Complex { real: value, imag: 0.0 }
    }
}


// ---------------- Unit tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Complex { real: 1.0, imag: 2.0 };
        let b = Complex { real: 3.0, imag: 4.0 };
        assert_eq!(a + b, Complex { real: 4.0, imag: 6.0 });
    }

    #[test]
    fn test_sub() {
        let a = Complex { real: 5.0, imag: 6.0 };
        let b = Complex { real: 2.0, imag: 3.0 };
        assert_eq!(a - b, Complex { real: 3.0, imag: 3.0 });
    }

    #[test]
    fn test_mul() {
        let a = Complex { real: 1.0, imag: 2.0 };
        let b = Complex { real: 3.0, imag: 4.0 };
        assert_eq!(a * b, Complex { real: -5.0, imag: 10.0 });
    }

    // Additional tests for mixed real and complex operations

    #[test]
    fn test_add_real() {
        let a = Complex { real: 1.0, imag: 2.0 };
        assert_eq!(a + 3.0, Complex { real: 4.0, imag: 2.0 });
    }

    #[test]
    fn test_sub_real() {
        let a = Complex { real: 5.0, imag: 6.0 };
        assert_eq!(a - 2.0, Complex { real: 3.0, imag: 6.0 });
    }

    #[test]
    fn test_mul_real() {
        let a = Complex { real: 1.0, imag: 2.0 };
        assert_eq!(a * 3.0, Complex { real: 3.0, imag: 6.0 });
    }
}