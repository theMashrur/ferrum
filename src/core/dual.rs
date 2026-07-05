use super::elementary_functions::ElementaryFunctions;
use std::fmt;
use std::ops;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub real: f64,
    pub dual: f64,
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dual >= 0.0 {
            write!(f, "{} + {}e", self.real, self.dual)
        } else {
            write!(f, "{} - {}e", self.real, self.dual)
        }
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Dual {
            real: value,
            dual: 0.0,
        }
    }
}

// ----------------- Arithmetic ---------------------

impl ops::Add<Dual> for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Dual {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl ops::Sub<Dual> for Dual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Dual {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl ops::Mul<Dual> for Dual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Dual {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl ElementaryFunctions for Dual {
    fn exp(self) -> Self {
        let exp_real = self.real.exp();
        Dual {
            real: exp_real,
            dual: exp_real * self.dual,
        }
    }

    fn ln(self) -> Self {
        Dual {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }

    fn sqrt(self) -> Self {
        let sqrt_real = self.real.sqrt();
        Dual {
            real: sqrt_real,
            dual: self.dual / (2.0 * sqrt_real),
        }
    }

    fn powf(self, n: f64) -> Self {
        let pow_real = self.real.powf(n);
        Dual {
            real: pow_real,
            dual: n * self.real.powf(n - 1.0) * self.dual,
        }
    }

    fn sin(self) -> Self {
        Dual {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }

    fn cos(self) -> Self {
        Dual {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }

    fn tan(self) -> Self {
        let cos_real = self.real.cos();
        Dual {
            real: self.real.tan(),
            dual: self.dual / (cos_real * cos_real),
        }
    }

    fn sinh(self) -> Self {
        Dual {
            real: self.real.sinh(),
            dual: self.dual * self.real.cosh(),
        }
    }

    fn cosh(self) -> Self {
        Dual {
            real: self.real.cosh(),
            dual: self.dual * self.real.sinh(),
        }
    }
}

// ------------------ Mixed Arithmetic ---------------------

impl ops::Add<f64> for Dual {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Dual {
            real: self.real + rhs,
            dual: self.dual,
        }
    }
}

impl ops::Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Dual {
            real: self.real - rhs,
            dual: self.dual,
        }
    }
}

impl ops::Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Dual {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

// ----------------- Tests ---------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}, tol {EPSILON}"
        );
    }

    fn assert_dual_close(actual: Dual, expected: Dual) {
        assert_close(actual.real, expected.real);
        assert_close(actual.dual, expected.dual);
    }

    #[test]
    fn test_addition() {
        let a = Dual {
            real: 2.0,
            dual: 3.0,
        };
        let b = Dual {
            real: 4.0,
            dual: 5.0,
        };
        let result = a + b;
        assert_eq!(
            result,
            Dual {
                real: 6.0,
                dual: 8.0
            }
        );
    }

    #[test]
    fn test_subtraction() {
        let a = Dual {
            real: 5.0,
            dual: 7.0,
        };
        let b = Dual {
            real: 2.0,
            dual: 3.0,
        };
        let result = a - b;
        assert_eq!(
            result,
            Dual {
                real: 3.0,
                dual: 4.0
            }
        );
    }

    #[test]
    fn test_multiplication() {
        let a = Dual {
            real: 2.0,
            dual: 3.0,
        };
        let b = Dual {
            real: 4.0,
            dual: 5.0,
        };
        let result = a * b;
        assert_eq!(
            result,
            Dual {
                real: 8.0,
                dual: 22.0
            }
        );
    }

    #[test]
    fn test_pow() {
        let a = Dual {
            real: 2.0,
            dual: 3.0,
        };
        let result = a.powf(3.0);
        assert_eq!(
            result,
            Dual {
                real: 8.0,
                dual: 36.0
            }
        );
    }

    #[test]
    fn test_exp() {
        let a = Dual {
            real: 1.0,
            dual: 2.0,
        };
        let result = a.exp();
        assert_eq!(
            result,
            Dual {
                real: std::f64::consts::E,
                dual: 2.0 * std::f64::consts::E
            }
        );
    }

    #[test]
    fn test_ln() {
        let a = Dual {
            real: std::f64::consts::E,
            dual: 2.0,
        };
        let result = a.ln();
        assert_eq!(
            result,
            Dual {
                real: 1.0,
                dual: 2.0 / std::f64::consts::E
            }
        );
    }

    #[test]
    fn test_sqrt() {
        let a = Dual {
            real: 4.0,
            dual: 2.0,
        };
        let result = a.sqrt();
        assert_eq!(
            result,
            Dual {
                real: 2.0,
                dual: 0.5
            }
        );
    }

    #[test]
    fn test_sin() {
        let a = Dual {
            real: std::f64::consts::PI / 2.0,
            dual: 1.0,
        };
        let result = a.sin();
        assert_dual_close(
            result,
            Dual {
                real: 1.0,
                dual: 0.0,
            },
        );
    }

    #[test]
    fn test_cos() {
        let a = Dual {
            real: std::f64::consts::PI,
            dual: 1.0,
        };
        let result = a.cos();
        assert_dual_close(
            result,
            Dual {
                real: -1.0,
                dual: 0.0,
            },
        );
    }

    #[test]
    fn test_tan() {
        let a = Dual {
            real: std::f64::consts::PI / 4.0,
            dual: 1.0,
        };
        let result = a.tan();
        assert_dual_close(
            result,
            Dual {
                real: 1.0,
                dual: 2.0,
            },
        );
    }

    #[test]
    fn test_sinh() {
        let a = Dual {
            real: 0.0,
            dual: 1.0,
        };
        let result = a.sinh();
        assert_eq!(
            result,
            Dual {
                real: 0.0,
                dual: 1.0
            }
        );
    }

    #[test]
    fn test_cosh() {
        let a = Dual {
            real: 0.0,
            dual: 1.0,
        };
        let result = a.cosh();
        assert_eq!(
            result,
            Dual {
                real: 1.0,
                dual: 0.0
            }
        );
    }
}
