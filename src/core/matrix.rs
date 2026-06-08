use std::fmt;
use std::ops;
use std::cmp;
use super::complex::Complex;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

// Concrete aliases used throughout the project.
pub type RealMatrix = Matrix<f64>;
pub type ComplexMatrix = Matrix<Complex>;

// Pretty-printer for matrices with NumPy-like truncation for large shapes.
impl<T> fmt::Display for Matrix<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const EDGE_ITEMS: usize = 2;

        let visible_indices = |len: usize| -> (Vec<usize>, bool) {
            if len <= EDGE_ITEMS * 2 {
                ((0..len).collect(), false)
            } else {
                let mut indices: Vec<usize> = (0..EDGE_ITEMS).collect();
                indices.extend((len - EDGE_ITEMS)..len);
                (indices, true)
            }
        };

        let (row_indices, rows_truncated) = visible_indices(self.rows);
        let (col_indices, cols_truncated) = visible_indices(self.cols);

        write!(f, "[")?;

        for (row_pos, &row) in row_indices.iter().enumerate() {
            if row_pos > 0 {
                writeln!(f)?;
            }

            if rows_truncated && row_pos == EDGE_ITEMS {
                writeln!(f, " ...")?;
            }

            let mut row_data: Vec<String> = Vec::new();
            for (col_pos, &col) in col_indices.iter().enumerate() {
                if cols_truncated && col_pos == EDGE_ITEMS {
                    row_data.push("...".to_string());
                }
                row_data.push(format!("{}", self.data[row * self.cols + col]));
            }

            if row_pos == 0 {
                write!(f, "[{}]", row_data.join(", "))?;
            } else {
                write!(f, " [{}]", row_data.join(", "))?;
            }
        }

        write!(f, "]")
    }
}

// ---------- Addition ----------

// Owned + owned matrix addition for the same scalar type.
impl<T> ops::Add<Matrix<T>> for Matrix<T>  where T: ops::Add<Output = T> + Copy {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data = self.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// Borrowed + borrowed matrix addition for the same scalar type.
impl<T> ops::Add<&Matrix<T>> for &Matrix<T>  where T: ops::Add<Output = T> + Copy {
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data = self.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// Real + complex promotes to complex.
impl ops::Add<Matrix<Complex>> for Matrix<f64> {
    type Output = ComplexMatrix;

    fn add(self, rhs: Matrix<Complex>) -> Matrix<Complex> { 
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data = self.data.iter().zip(rhs.data.iter())
        .map(|(a, b)| Complex { real: *a + b.real, imag: b.imag}).collect();
        ComplexMatrix { rows: self.rows, cols: self.cols, data }
    }
}

// Borrowed real + borrowed complex promotes to complex.
impl ops::Add<&Matrix<Complex>> for &Matrix<f64> {
    type Output = ComplexMatrix;

    fn add(self, rhs: &Matrix<Complex>) -> Matrix<Complex> { 
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data = self.data.iter().zip(rhs.data.iter())
        .map(|(a, b)| Complex { real: *a + b.real, imag: b.imag}).collect();
        ComplexMatrix { rows: self.rows, cols: self.cols, data }
    }
}

// Complex + real delegates to the real + complex implementation.
impl ops::Add<Matrix<f64>> for Matrix<Complex> {
    type Output = Matrix<Complex>;

    fn add(self, rhs: Matrix<f64>) -> Matrix<Complex> { 
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        rhs + self
    }
}

// Borrowed complex + borrowed real delegates to the borrowed real + borrowed complex implementation.
impl ops::Add<&Matrix<f64>> for &Matrix<Complex> {
    type Output = Matrix<Complex>;

    fn add(self, rhs: &Matrix<f64>) -> Matrix<Complex> { 
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        rhs + self
    }
}

// ---------- Scalar Multiplication ----------

// Owned matrix by real scalar multiplication.
impl<T> ops::Mul<f64> for Matrix<T> where T: ops::Mul<f64, Output = T> + Copy {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let data = self.data.iter()
            .map(|x| *x * rhs)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// Borrowed matrix by real scalar multiplication.
impl<T> ops::Mul<f64> for &Matrix<T> where T: ops::Mul<f64, Output = T> + Copy {
    type Output = Matrix<T>;

    fn mul(self, rhs: f64) -> Self::Output {
        let data = self.data.iter()
            .map(|x| *x * rhs)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// Owned matrix by complex scalar multiplication.
impl<T> ops::Mul<Complex> for Matrix<T> where T: ops::Mul<Complex, Output = T> + Copy {
    type Output = Self;

    fn mul(self, rhs: Complex) -> Self::Output {
        let data = self.data.iter()
            .map(|x| *x * rhs)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// Borrowed matrix by complex scalar multiplication.
impl<T> ops::Mul<Complex> for &Matrix<T> where T: ops::Mul<Complex, Output = T> + Copy {
    type Output = Matrix<T>;

    fn mul(self, rhs: Complex) -> Self::Output {
        let data = self.data.iter()
            .map(|x| *x * rhs)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

// ---------- Subtraction ----------

// Owned - owned implemented via addition and scalar multiplication.
impl<T> ops::Sub<Matrix<T>> for Matrix<T>
where
    T: ops::Sub<Output = T> + ops::Mul<f64, Output = T> + ops::Add<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        self + (rhs * -1.0)
    }
}

// Borrowed - borrowed implemented via addition and scalar multiplication.
impl<T> ops::Sub<&Matrix<T>> for &Matrix<T>
where
    T: ops::Sub<Output = T> + ops::Mul<f64, Output = T> + ops::Add<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        self + &(rhs * -1.0)
    }
}

impl ops::Sub<Matrix<Complex>> for Matrix<f64> {
    type Output = ComplexMatrix;

    fn sub(self, rhs: Matrix<Complex>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        self + (rhs * -1.0)
    }
}

impl ops::Sub<Matrix<f64>> for Matrix<Complex> {
    type Output = ComplexMatrix;

    fn sub(self, rhs: Matrix<f64>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        (rhs * -1.0) + self
    }
}

impl ops::Sub<&Matrix<Complex>> for &Matrix<f64> {
    type Output = ComplexMatrix;

    fn sub(self, rhs: &Matrix<Complex>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        self + &(rhs * -1.0)
    }
}

impl ops::Sub<&Matrix<f64>> for &Matrix<Complex> {
    type Output = ComplexMatrix;

    fn sub(self, rhs: &Matrix<f64>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        &(rhs * -1.0) + self
    }
}

// ---------- Constructors ----------

impl<T> Matrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len(), "Data length must match rows * cols");
        Matrix { rows, cols, data }
    }
}

// ---------- Utilities (Real Matrices) ----------

impl Matrix<f64> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut data = Self::zeros(size, size).data;
        for i in 0..size { 
            data[i * size + i] = 1.0;
        }
        Matrix { rows: size, cols: size, data}
    }
}

// ---------- Utilities (All Matrices) ----------

impl<T> Matrix<T> {
    pub fn transpose(&self, blocksize: Option<usize>) -> Self
    where 
        T: Copy,
    {
        let blocksize = blocksize.unwrap_or(32);
        let mut transposed_data = vec![self.data[0]; self.rows * self.cols];
        for i in (0..self.rows).step_by(blocksize) {
            for j in (0..self.cols).step_by(blocksize) {
                let i_max = cmp::min(i+blocksize, self.rows);
                let j_max = cmp::min(j+blocksize, self.cols);
                for ii in i..i_max {
                    for jj in j..j_max {
                        transposed_data[jj * self.rows + ii] = self.data[ii * self.cols + jj];
                    }
                }
            }
        }
        Matrix { rows: self.cols, cols: self.rows, data: transposed_data }
    }
}
// ---------- Conversions ----------

// Promote a real matrix to complex by adding zero imaginary parts.
impl From<Matrix<f64>> for Matrix<Complex> {
    fn from(matrix: Matrix<f64>) -> Self {
        let data = matrix.data.into_iter()
            .map(|x| Complex::from(x))
            .collect();
        Matrix { rows: matrix.rows, cols: matrix.cols, data }
    }
}

// ----------- Unit tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    // Test cases for real-valued matrix operations.
    #[test]
    fn test_real_matrix_addition() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a + b;
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_real_matrix_borrowed_addition() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = &a + &b;
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_real_matrix_subtraction() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a - b;
        assert_eq!(c.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_real_matrix_borrowed_subtraction() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a - &b;
        assert_eq!(c.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_real_matrix_scalar_multiplication() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a * 2.0;
        assert_eq!(c.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    // Test cases for complex-valued matrix operations

    #[test]
    fn test_complex_matrix_addition() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c: ComplexMatrix = a + b;
        assert_eq!(c.data, vec![
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 8.0, imag: 8.0 },
            Complex { real: 10.0, imag: 10.0 },
            Complex { real: 12.0, imag: 12.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_borrowed_addition() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c = &a + &b;
        assert_eq!(c.data, vec![
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 8.0, imag: 8.0 },
            Complex { real: 10.0, imag: 10.0 },
            Complex { real: 12.0, imag: 12.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_subtraction() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let c = a - b;
        assert_eq!(c.data, vec![
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_borrowed_subtraction() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let c = &a - &b;
        assert_eq!(c.data, vec![
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_scalar_multiplication() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let c = a * 2.0;
        assert_eq!(c.data, vec![
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 4.0, imag: 4.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_scalar_multiplication_complex() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let c = a * Complex { real: 2.0, imag: 3.0 };
        assert_eq!(c.data, vec![
            Complex { real: -1.0, imag: 5.0 },
            Complex { real: -2.0, imag: 10.0 },
            Complex { real: -3.0, imag: 15.0 },
            Complex { real: -4.0, imag: 20.0 },
        ]);
    }

    #[test]
    fn test_complex_matrix_borrowed_scalar_multiplication_complex() {
        let a: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 1.0, imag: 1.0 },
            Complex { real: 2.0, imag: 2.0 },
            Complex { real: 3.0, imag: 3.0 },
            Complex { real: 4.0, imag: 4.0 },
        ]);
        let c = &a * Complex { real: 2.0, imag: 3.0 };
        assert_eq!(c.data, vec![
            Complex { real: -1.0, imag: 5.0 },
            Complex { real: -2.0, imag: 10.0 },
            Complex { real: -3.0, imag: 15.0 },
            Complex { real: -4.0, imag: 20.0 },
        ]);
    }

    // Additional tests for mixed real-complex operations

    #[test]
    fn test_real_plus_complex() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c = a + b;
        assert_eq!(c.data, vec![
            Complex { real: 6.0, imag: 5.0 },
            Complex { real: 8.0, imag: 6.0 },
            Complex { real: 10.0, imag: 7.0 },
            Complex { real: 12.0, imag: 8.0 },
        ]);

        let d = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let e = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let f = e + d;
        assert_eq!(f.data, vec![
            Complex { real: 6.0, imag: 5.0 },
            Complex { real: 8.0, imag: 6.0 },
            Complex { real: 10.0, imag: 7.0 },
            Complex { real: 12.0, imag: 8.0 },
        ]);
    }

    #[test]
    fn test_borrowed_real_plus_complex() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c = &a + &b;
        assert_eq!(c.data, vec![
            Complex { real: 6.0, imag: 5.0 },
            Complex { real: 8.0, imag: 6.0 },
            Complex { real: 10.0, imag: 7.0 },
            Complex { real: 12.0, imag: 8.0 },
        ]);

        let d = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let e = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let f = &e + &d;
        assert_eq!(f.data, vec![
            Complex { real: 6.0, imag: 5.0 },
            Complex { real: 8.0, imag: 6.0 },
            Complex { real: 10.0, imag: 7.0 },
            Complex { real: 12.0, imag: 8.0 },
        ]);
    }

    #[test]
    fn test_real_minus_complex() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c = a - b;
        assert_eq!(c.data, vec![
            Complex { real: -4.0, imag: -5.0 },
            Complex { real: -4.0, imag: -6.0 },
            Complex { real: -4.0, imag: -7.0 },
            Complex { real: -4.0, imag: -8.0 },
        ]);
        let d = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let e = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let f = e - d;
        assert_eq!(f.data, vec![
            Complex { real: 4.0, imag: 5.0 },
            Complex { real: 4.0, imag: 6.0 },
            Complex { real: 4.0, imag: 7.0 },
            Complex { real: 4.0, imag: 8.0 },
        ]);
    }

    #[test]
    fn test_borrowed_real_minus_complex() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: ComplexMatrix = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let c = &a - &b;
        assert_eq!(c.data, vec![
            Complex { real: -4.0, imag: -5.0 },
            Complex { real: -4.0, imag: -6.0 },
            Complex { real: -4.0, imag: -7.0 },
            Complex { real: -4.0, imag: -8.0 },
        ]);
        let d = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let e = ComplexMatrix::new(2, 2, vec![
            Complex { real: 5.0, imag: 5.0 },
            Complex { real: 6.0, imag: 6.0 },
            Complex { real: 7.0, imag: 7.0 },
            Complex { real: 8.0, imag: 8.0 },
        ]);
        let f = &e - &d;
        assert_eq!(f.data, vec![
            Complex { real: 4.0, imag: 5.0 },
            Complex { real: 4.0, imag: 6.0 },
            Complex { real: 4.0, imag: 7.0 },
            Complex { real: 4.0, imag: 8.0 },
        ]);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.transpose(None);
        assert_eq!(b.rows, 3);
        assert_eq!(b.cols, 2);
        assert_eq!(b.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}