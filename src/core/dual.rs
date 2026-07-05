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

impl Dual {
    fn pow(self, power: f64) -> Self {
        Dual {
            real: self.real.powf(power),
            dual: power * self.real.powf(power - 1.0) * self.dual,
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
        let result = a.pow(3.0);
        assert_eq!(
            result,
            Dual {
                real: 8.0,
                dual: 36.0
            }
        );
    }
}
