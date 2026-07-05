use super::elementary_functions::ElementaryFunctions;

impl ElementaryFunctions for f64 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }
}
