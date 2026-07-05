use std::cmp;
use std::fmt;
use std::ops;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

// Central Trait
pub trait MatrixRead<T> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> &T;
}

pub type RealMatrix = Matrix<f64>;

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

impl MatrixRead<f64> for RealMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get(&self, row: usize, col: usize) -> &f64 {
        &self.data[row * self.cols + col]
    }
}

// ---------- Constructors ----------

impl<T> Matrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data length must match rows * cols"
        );
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
        Matrix {
            rows: size,
            cols: size,
            data,
        }
    }
}

// ---------- Utilities (All Matrices) ----------

impl<T> Matrix<T> {
    pub fn transpose(&self, blocksize: Option<usize>) -> Self
    where
        T: Copy,
    {
        if self.rows == 0 || self.cols == 0 {
            return Matrix {
                rows: self.cols,
                cols: self.rows,
                data: vec![],
            };
        }
        let blocksize = blocksize.unwrap_or(32);
        let mut transposed_data = vec![self.data[0]; self.rows * self.cols];
        for i in (0..self.rows).step_by(blocksize) {
            for j in (0..self.cols).step_by(blocksize) {
                let i_max = cmp::min(i + blocksize, self.rows);
                let j_max = cmp::min(j + blocksize, self.cols);
                for ii in i..i_max {
                    for jj in j..j_max {
                        transposed_data[jj * self.rows + ii] = self.data[ii * self.cols + jj];
                    }
                }
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }
}

// ---------- Arithmetic Operations ----------

impl<'a, 'b, T> ops::Add<&'b Matrix<T>> for &'a Matrix<T>
where
    T: Copy + ops::Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let new_data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl<'a, 'b, T> ops::Sub<&'b Matrix<T>> for &'a Matrix<T>
where
    T: Copy + ops::Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let new_data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl<'a, T> ops::Mul<T> for &'a Matrix<T>
where
    T: Copy + ops::Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Matrix<T> {
        let new_data = self.data.iter().map(|&x| x * rhs).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

// ---------- Unit Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(*m.get(1, 2), 6.0);
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m3 = &m1 + &m2;
        assert_eq!(m3.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m2 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m3 = &m1 - &m2;
        assert_eq!(m3.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m_scaled = &m * 2.0;
        assert_eq!(m_scaled.data, vec![2.0, 4.0, 6.0, 8.0]);
    }
}
