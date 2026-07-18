use std::cmp;
use std::fmt;
use std::ops;
use std::ops::Range;

use super::views::{ColView, ColViewMut, MatrixView, MatrixViewMut, RowView, RowViewMut};

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

    fn is_row_contiguous(&self) -> bool;
    fn row(&self, row: usize) -> RowView<'_, T>;
    fn col(&self, col: usize) -> ColView<'_, T>;
}

pub trait MatrixWrite<T>: MatrixRead<T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T;

    fn accumulate(&mut self, row: usize, col: usize, value: T)
    where
        T: ops::AddAssign,
    {
        let elem = self.get_mut(row, col);
        *elem += value;
    }

    fn row_mut(&mut self, row: usize) -> RowViewMut<'_, T>;
    fn col_mut(&mut self, col: usize) -> ColViewMut<'_, T>;
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

impl<'a, T> MatrixRead<T> for Matrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    fn is_row_contiguous(&self) -> bool {
        true
    }

    fn row(&self, row: usize) -> RowView<'_, T> {
        assert!(row < self.rows);
        RowView {
            cols: self.cols,
            data: &self.data,
            offset: row * self.cols,
            col_stride: 1,
        }
    }

    fn col(&self, col: usize) -> ColView<'_, T> {
        assert!(col < self.cols);
        ColView {
            rows: self.rows,
            data: &self.data,
            offset: col,
            row_stride: self.cols,
        }
    }
}

impl<T> MatrixWrite<T> for Matrix<T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }

    fn row_mut(&mut self, row: usize) -> RowViewMut<'_, T> {
        assert!(row < self.rows);
        RowViewMut {
            cols: self.cols,
            data: &mut self.data,
            offset: row * self.cols,
            col_stride: 1,
        }
    }

    fn col_mut(&mut self, col: usize) -> ColViewMut<'_, T> {
        assert!(col < self.cols);
        ColViewMut {
            rows: self.rows,
            data: &mut self.data,
            offset: col,
            row_stride: self.cols,
        }
    }
}

// ---------- Constructors ----------

impl<T> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self
    where
        T: From<f64> + Copy,
    {
        Self::zeros(rows, cols)
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data length must match rows * cols"
        );
        Matrix { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: From<f64> + Copy,
    {
        Matrix {
            rows,
            cols,
            data: vec![T::from(0.0); rows * cols],
        }
    }
}

// ---------- Utilities (Real Matrices) ----------

impl Matrix<f64> {
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

pub fn add_core<T>(lhs: &Matrix<T>, rhs: &Matrix<T>, out: &mut Matrix<T>)
where
    T: Copy + ops::Add<Output = T>,
{
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    assert_eq!(lhs.rows, out.rows);
    assert_eq!(lhs.cols, out.cols);

    for i in 0..lhs.rows {
        for j in 0..lhs.cols {
            out.data[i * lhs.cols + j] = lhs.data[i * lhs.cols + j] + rhs.data[i * lhs.cols + j];
        }
    }
}

impl<'a, 'b, T> ops::Add<&'b Matrix<T>> for &'a Matrix<T>
where
    T: Copy + ops::Add<Output = T> + From<f64>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        let mut out = Matrix::new(self.rows, self.cols);
        add_core(self, rhs, &mut out);
        out
    }
}

pub fn sub_core<T>(lhs: &Matrix<T>, rhs: &Matrix<T>, out: &mut Matrix<T>)
where
    T: Copy + ops::Sub<Output = T>,
{
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    assert_eq!(lhs.rows, out.rows);
    assert_eq!(lhs.cols, out.cols);

    for i in 0..lhs.rows {
        for j in 0..lhs.cols {
            out.data[i * lhs.cols + j] = lhs.data[i * lhs.cols + j] - rhs.data[i * lhs.cols + j];
        }
    }
}

impl<'a, 'b, T> ops::Sub<&'b Matrix<T>> for &'a Matrix<T>
where
    T: Copy + ops::Sub<Output = T> + From<f64>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        let mut out = Matrix::new(self.rows, self.cols);
        sub_core(self, rhs, &mut out);
        out
    }
}

pub fn scalar_mul_core<T>(matrix: &Matrix<T>, scalar: T, out: &mut Matrix<T>)
where
    T: Copy + ops::Mul<Output = T>,
{
    assert_eq!(matrix.rows, out.rows);
    assert_eq!(matrix.cols, out.cols);

    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            out.data[i * matrix.cols + j] = matrix.data[i * matrix.cols + j] * scalar;
        }
    }
}

impl<'a, T> ops::Mul<T> for &'a Matrix<T>
where
    T: Copy + ops::Mul<Output = T> + From<f64>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Matrix<T> {
        let mut out = Matrix::new(self.rows, self.cols);
        scalar_mul_core(self, rhs, &mut out);
        out
    }
}

pub fn gemm_kernel<A, B, C, T>(a: &A, b: &B, out: &mut C, alpha: Option<T>, beta: Option<T>)
where
    A: MatrixRead<T>,
    B: MatrixRead<T>,
    C: MatrixWrite<T>,
    T: Copy
        + ops::Add<Output = T>
        + ops::Mul<Output = T>
        + Default
        + PartialEq
        + From<f64>
        + ops::AddAssign<T>
        + ops::Sub<Output = T>,
{
    assert_eq!(
        a.cols(),
        b.rows(),
        "Inner dimensions {} and {} must match for matrix multiplication",
        a.cols(),
        b.rows()
    );
    let m = a.rows();
    let k_dim = a.cols();
    let n = b.cols();
    let beta = beta.unwrap_or(T::from(0.0));
    let alpha = alpha.unwrap_or(T::from(1.0));

    // beta scaling block: skip if beta is zero
    if beta != T::from(1.0) {
        for i in 0..m {
            for j in 0..n {
                out.accumulate(i, j, (beta - T::from(1.0)) * (*out.get(i, j)));
            }
        }
    }

    // Matrix Multiplication block: skip if alpha is zero
    if alpha != T::from(0.0) {
        for i in 0..m {
            for k in 0..k_dim {
                let a_ik = a.get(i, k);
                for j in 0..n {
                    let b_kj = b.get(k, j);
                    out.accumulate(i, j, (*a_ik) * (*b_kj) * alpha);
                }
            }
        }
    }
}

impl<'a, 'b, T> ops::Mul<&'b Matrix<T>> for &'a Matrix<T>
where
    T: Copy
        + ops::Add<Output = T>
        + ops::Mul<Output = T>
        + Default
        + PartialEq
        + From<f64>
        + ops::AddAssign<T>
        + ops::Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.cols, rhs.rows,
            "Inner dimensions must match for matrix multiplication"
        );
        let mut out = Matrix::zeros(self.rows, rhs.cols);
        gemm_kernel(self, rhs, &mut out, None, None);
        out
    }
}

// ---------- Views and Indexing -------------

impl<T> Matrix<T> {
    pub fn view(&self, row_range: Range<usize>, col_range: Range<usize>) -> MatrixView<'_, T> {
        assert!(row_range.start <= row_range.end && row_range.end <= self.rows);
        assert!(col_range.start <= col_range.end && col_range.end <= self.cols);

        MatrixView {
            rows: row_range.end - row_range.start,
            cols: col_range.end - col_range.start,
            data: &self.data,
            offset: row_range.start * self.cols + col_range.start,
            row_stride: self.cols,
            col_stride: 1,
        }
    }

    pub fn view_mut(
        &mut self,
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) -> MatrixViewMut<'_, T> {
        assert!(row_range.start <= row_range.end && row_range.end <= self.rows);
        assert!(col_range.start <= col_range.end && col_range.end <= self.cols);

        MatrixViewMut {
            rows: row_range.end - row_range.start,
            cols: col_range.end - col_range.start,
            data: &mut self.data,
            offset: row_range.start * self.cols + col_range.start,
            row_stride: self.cols,
            col_stride: 1,
        }
    }

    pub fn row_view(&self, row: usize) -> RowView<'_, T> {
        assert!(row < self.rows);
        RowView {
            cols: self.cols,
            data: &self.data,
            offset: row * self.cols,
            col_stride: 1,
        }
    }

    pub fn row_view_mut(&mut self, row: usize) -> RowViewMut<'_, T> {
        assert!(row < self.rows);
        RowViewMut {
            cols: self.cols,
            data: &mut self.data,
            offset: row * self.cols,
            col_stride: 1,
        }
    }

    pub fn col_view(&self, col: usize) -> ColView<'_, T> {
        assert!(col < self.cols);
        ColView {
            rows: self.rows,
            data: &self.data,
            offset: col,
            row_stride: self.cols,
        }
    }

    pub fn col_view_mut(&mut self, col: usize) -> ColViewMut<'_, T> {
        assert!(col < self.cols);
        ColViewMut {
            rows: self.rows,
            data: &mut self.data,
            offset: col,
            row_stride: self.cols,
        }
    }
}

// ---------- Unit Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(*m.get(1, 2), 6.0);
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m3 = &m1 + &m2;
        assert_eq!(m3.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m2 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m3 = &m1 - &m2;
        assert_eq!(m3.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m_scaled = &m * 2.0;
        assert_eq!(m_scaled.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_new_initializes_with_zeros() {
        let m: Matrix<f64> = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data, vec![0.0; 6]);
    }

    #[test]
    fn test_matrix_view() {
        let m = Matrix::from_data(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let view = m.view(1..3, 1..3);
        assert_eq!(view.rows(), 2);
        assert_eq!(view.cols(), 2);
        assert_eq!(*view.get(0, 0), 6.0);
        assert_eq!(*view.get(1, 1), 11.0);
    }

    #[test]
    fn test_matrix_row_view() {
        let m = Matrix::from_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let row_view = m.row_view(1);
        assert_eq!(row_view.cols, 3);
        assert_eq!(*row_view.get(0, 0), 4);
        assert_eq!(*row_view.get(0, 2), 6);
    }

    #[test]
    fn test_matrix_col_view() {
        let m = Matrix::from_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let col_view = m.col_view(1);
        assert_eq!(col_view.rows, 3);
        assert_eq!(*col_view.get(0, 0), 2);
        assert_eq!(*col_view.get(2, 0), 8);
    }

    #[test]
    fn test_matrix_read_col_on_matrix() {
        let m = Matrix::from_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let col = m.col(1);

        assert_eq!(col.rows(), 3);
        assert_eq!(col.cols(), 1);
        assert_eq!(*col.get(0, 0), 2);
        assert_eq!(*col.get(2, 0), 8);
    }

    #[test]
    fn test_matrix_read_col_on_matrix_view() {
        let m = Matrix::from_data(
            4,
            4,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        let view = m.view(1..4, 1..3);
        let col = view.col(1);

        assert_eq!(col.rows(), 3);
        assert_eq!(col.cols(), 1);
        assert_eq!(*col.get(0, 0), 7);
        assert_eq!(*col.get(2, 0), 15);
    }

    #[test]
    fn test_matrix_read_col_on_row_view() {
        let m = Matrix::from_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let row = m.row(1);
        let col = row.col(2);

        assert_eq!(col.rows(), 1);
        assert_eq!(col.cols(), 1);
        assert_eq!(*col.get(0, 0), 6);
    }

    #[test]
    fn test_matrix_read_col_on_col_view() {
        let m = Matrix::from_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let first_col = m.col(1);
        let nested_col = first_col.col(0);

        assert_eq!(nested_col.rows(), 3);
        assert_eq!(nested_col.cols(), 1);
        assert_eq!(*nested_col.get(1, 0), 5);
    }

    #[test]
    fn test_gemm_kernel_basic() {
        let a = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = Matrix::zeros(2, 2);

        gemm_kernel(&a, &b, &mut out, None, None);

        // Manual calculation: [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(*out.get(0, 0), 58.0);
        assert_eq!(*out.get(0, 1), 64.0);
        assert_eq!(*out.get(1, 0), 139.0);
        assert_eq!(*out.get(1, 1), 154.0);
    }

    #[test]
    fn test_gemm_kernel_with_alpha() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![2.0, 0.0, 1.0, 2.0]);
        let mut out = Matrix::zeros(2, 2);

        gemm_kernel(&a, &b, &mut out, Some(2.0), None);

        // Expected: 2 * (A * B) = 2 * [[4, 4], [10, 8]]
        assert_eq!(*out.get(0, 0), 8.0);
        assert_eq!(*out.get(0, 1), 8.0);
        assert_eq!(*out.get(1, 0), 20.0);
        assert_eq!(*out.get(1, 1), 16.0);
    }

    #[test]
    fn test_gemm_kernel_with_beta() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![2.0, 0.0, 1.0, 2.0]);
        let mut out = Matrix::from_data(2, 2, vec![1.0, 1.0, 1.0, 1.0]);

        gemm_kernel(&a, &b, &mut out, None, Some(2.0));

        // Expected: beta * out + A * B = 2*out + A*B = [[6, 6], [12, 10]]
        assert_eq!(*out.get(0, 0), 6.0);
        assert_eq!(*out.get(0, 1), 6.0);
        assert_eq!(*out.get(1, 0), 12.0);
        assert_eq!(*out.get(1, 1), 10.0);
    }

    #[test]
    fn test_gemm_kernel_with_alpha_and_beta() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![2.0, 0.0, 1.0, 2.0]);
        let mut out = Matrix::from_data(2, 2, vec![1.0, 1.0, 1.0, 1.0]);

        gemm_kernel(&a, &b, &mut out, Some(3.0), Some(2.0));

        // Expected: beta * out + alpha * A * B = 2*out + 3*(A*B) = [[14, 14], [32, 26]]
        assert_eq!(*out.get(0, 0), 14.0);
        assert_eq!(*out.get(0, 1), 14.0);
        assert_eq!(*out.get(1, 0), 32.0);
        assert_eq!(*out.get(1, 1), 26.0);
    }

    #[test]
    fn test_gemm_kernel_zero_alpha() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![2.0, 0.0, 1.0, 2.0]);
        let mut out = Matrix::from_data(2, 2, vec![5.0, 5.0, 5.0, 5.0]);

        gemm_kernel(&a, &b, &mut out, Some(0.0), Some(2.0));

        // Expected: only beta scaling, so 2*out
        assert_eq!(*out.get(0, 0), 10.0);
        assert_eq!(*out.get(0, 1), 10.0);
        assert_eq!(*out.get(1, 0), 10.0);
        assert_eq!(*out.get(1, 1), 10.0);
    }

    #[test]
    fn test_matmul_operator() {
        let a = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = &a * &b;

        // Manual calculation: [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(*c.get(0, 0), 58.0);
        assert_eq!(*c.get(0, 1), 64.0);
        assert_eq!(*c.get(1, 0), 139.0);
        assert_eq!(*c.get(1, 1), 154.0);
    }
}
