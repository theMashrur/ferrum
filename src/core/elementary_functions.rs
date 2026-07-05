pub trait ElementaryFunctions: Sized + Copy {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, n: f64) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
}

macro_rules! free_fn {
    ($name:ident) => {
        pub fn $name<T: ElementaryFunctions>(x: T) -> T {
            x.$name()
        }
    };
}

free_fn!(exp);
free_fn!(ln);
free_fn!(sqrt);
free_fn!(sin);
free_fn!(cos);
free_fn!(tan);
free_fn!(sinh);
free_fn!(cosh);

pub fn powf<T: ElementaryFunctions>(x: T, n: f64) -> T {
    x.powf(n)
}
