use super::elementary_functions::ElementaryFunctions;
use std::fmt;
use std::ops;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{} + {}i", self.real, self.imag)
        } else {
            write!(f, "{} - {}i", self.real, -self.imag)
        }
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

impl ops::AddAssign<Complex> for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
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
            imag: self.imag,
        }
    }
}

impl ops::Sub<f64> for Complex {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Complex {
            real: self.real - rhs,
            imag: self.imag,
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
            imag: self.imag * rhs,
        }
    }
}

impl From<f64> for Complex {
    fn from(value: f64) -> Self {
        Complex {
            real: value,
            imag: 0.0,
        }
    }
}

impl ElementaryFunctions for Complex {
    fn exp(self) -> Self {
        let exp_real = self.real.exp();
        Complex {
            real: exp_real * self.imag.cos(),
            imag: exp_real * self.imag.sin(),
        }
    }

    fn ln(self) -> Self {
        Complex {
            real: (self.real.powi(2) + self.imag.powi(2)).sqrt().ln(),
            imag: self.imag.atan2(self.real),
        }
    }

    fn sqrt(self) -> Self {
        let magnitude = (self.real.powi(2) + self.imag.powi(2)).sqrt();
        let angle = self.imag.atan2(self.real) / 2.0;
        Complex {
            real: magnitude.sqrt() * angle.cos(),
            imag: magnitude.sqrt() * angle.sin(),
        }
    }

    fn powf(self, n: f64) -> Self {
        let magnitude = (self.real.powi(2) + self.imag.powi(2)).sqrt().powf(n);
        let angle = self.imag.atan2(self.real) * n;
        Complex {
            real: magnitude * angle.cos(),
            imag: magnitude * angle.sin(),
        }
    }

    fn sin(self) -> Self {
        Complex {
            real: self.real.sin() * self.imag.cosh(),
            imag: self.real.cos() * self.imag.sinh(),
        }
    }

    fn cos(self) -> Self {
        Complex {
            real: self.real.cos() * self.imag.cosh(),
            imag: -self.real.sin() * self.imag.sinh(),
        }
    }

    fn tan(self) -> Self {
        let sin_val = self.sin();
        let cos_val = self.cos();
        Complex {
            real: sin_val.real / cos_val.real,
            imag: sin_val.imag / cos_val.real,
        }
    }

    fn sinh(self) -> Self {
        Complex {
            real: self.real.sinh() * self.imag.cos(),
            imag: self.real.cosh() * self.imag.sin(),
        }
    }

    fn cosh(self) -> Self {
        Complex {
            real: self.real.cosh() * self.imag.cos(),
            imag: self.real.sinh() * self.imag.sin(),
        }
    }
}

// ---------------- Unit tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Complex {
            real: 1.0,
            imag: 2.0,
        };
        let b = Complex {
            real: 3.0,
            imag: 4.0,
        };
        assert_eq!(
            a + b,
            Complex {
                real: 4.0,
                imag: 6.0
            }
        );
    }

    #[test]
    fn test_add_assign() {
        let mut value = Complex {
            real: 1.0,
            imag: 2.0,
        };

        value += Complex {
            real: 3.0,
            imag: 4.0,
        };

        assert_eq!(
            value,
            Complex {
                real: 4.0,
                imag: 6.0,
            }
        );
    }

    #[test]
    fn test_sub() {
        let a = Complex {
            real: 5.0,
            imag: 6.0,
        };
        let b = Complex {
            real: 2.0,
            imag: 3.0,
        };
        assert_eq!(
            a - b,
            Complex {
                real: 3.0,
                imag: 3.0
            }
        );
    }

    #[test]
    fn test_mul() {
        let a = Complex {
            real: 1.0,
            imag: 2.0,
        };
        let b = Complex {
            real: 3.0,
            imag: 4.0,
        };
        assert_eq!(
            a * b,
            Complex {
                real: -5.0,
                imag: 10.0
            }
        );
    }

    // Additional tests for mixed real and complex operations

    #[test]
    fn test_add_real() {
        let a = Complex {
            real: 1.0,
            imag: 2.0,
        };
        assert_eq!(
            a + 3.0,
            Complex {
                real: 4.0,
                imag: 2.0
            }
        );
    }

    #[test]
    fn test_sub_real() {
        let a = Complex {
            real: 5.0,
            imag: 6.0,
        };
        assert_eq!(
            a - 2.0,
            Complex {
                real: 3.0,
                imag: 6.0
            }
        );
    }

    #[test]
    fn test_mul_real() {
        let a = Complex {
            real: 1.0,
            imag: 2.0,
        };
        assert_eq!(
            a * 3.0,
            Complex {
                real: 3.0,
                imag: 6.0
            }
        );
    }
}
