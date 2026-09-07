use super::matrix::MatrixRead;
use super::matrix::MatrixWrite;
use std::fmt;

pub struct MatrixView<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

impl<'a, T> MatrixView<'a, T> {
    /// Contiguous slice for `row`, or `None` if the columns aren't unit-stride.
    pub fn row_slice(&self, row: usize) -> Option<&[T]> {
        assert!(row < self.rows, "Row index out of bounds");
        if self.col_stride != 1 {
            return None;
        }
        let start = self.offset + row * self.row_stride;
        Some(&self.data[start..start + self.cols])
    }
}

pub struct MatrixViewMut<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a mut [T],
    pub(crate) offset: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// Contiguous slice for `row`, or `None` if the columns aren't unit-stride.
    pub fn row_slice(&self, row: usize) -> Option<&[T]> {
        assert!(row < self.rows, "Row index out of bounds");
        if self.col_stride != 1 {
            return None;
        }
        let start = self.offset + row * self.row_stride;
        Some(&self.data[start..start + self.cols])
    }

    /// Mutable contiguous slice for `row`, or `None` if the columns aren't unit-stride.
    pub fn row_slice_mut(&mut self, row: usize) -> Option<&mut [T]> {
        assert!(row < self.rows, "Row index out of bounds");
        if self.col_stride != 1 {
            return None;
        }
        let start = self.offset + row * self.row_stride;
        Some(&mut self.data[start..start + self.cols])
    }
}

pub struct RowView<'a, T> {
    pub cols: usize,
    pub data: &'a [T],
    pub offset: usize,
    pub col_stride: usize,
}

impl<'a, T> RowView<'a, T> {
    /// The row as a contiguous slice, or `None` if it isn't unit-stride.
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.col_stride != 1 {
            return None;
        }
        Some(&self.data[self.offset..self.offset + self.cols])
    }
}

pub struct RowViewMut<'a, T> {
    pub cols: usize,
    pub data: &'a mut [T],
    pub offset: usize,
    pub col_stride: usize,
}

impl<'a, T> RowViewMut<'a, T> {
    /// The row as a contiguous slice, or `None` if it isn't unit-stride.
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.col_stride != 1 {
            return None;
        }
        Some(&self.data[self.offset..self.offset + self.cols])
    }

    /// The row as a mutable contiguous slice, or `None` if it isn't unit-stride.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.col_stride != 1 {
            return None;
        }
        Some(&mut self.data[self.offset..self.offset + self.cols])
    }
}

pub struct ColView<'a, T> {
    pub rows: usize,
    pub data: &'a [T],
    pub offset: usize,
    pub row_stride: usize,
}

pub struct ColViewMut<'a, T> {
    pub rows: usize,
    pub data: &'a mut [T],
    pub offset: usize,
    pub row_stride: usize,
}

macro_rules! impl_matrix_read_for_view {
    ($view_type:ident) => {
        impl<'a, T> MatrixRead<T> for $view_type<'a, T> {
            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                self.cols
            }

            fn get(&self, row: usize, col: usize) -> &T {
                let index = self.offset + row * self.row_stride + col * self.col_stride;
                &self.data[index]
            }

            fn is_row_contiguous(&self) -> bool {
                self.col_stride == 1
            }

            fn row(&self, row: usize) -> RowView<'_, T> {
                assert!(row < self.rows, "Row index out of bounds");
                RowView {
                    cols: self.cols,
                    data: self.data,
                    offset: self.offset + row * self.row_stride,
                    col_stride: self.col_stride,
                }
            }

            fn col(&self, col: usize) -> ColView<'_, T> {
                assert!(col < self.cols, "Column index out of bounds");
                ColView {
                    rows: self.rows,
                    data: self.data,
                    offset: self.offset + col * self.col_stride,
                    row_stride: self.row_stride,
                }
            }
        }
    };
}

macro_rules! impl_matrix_view_fmt {
    ($view_type:ident) => {
        impl<'a, T: fmt::Display> fmt::Display for $view_type<'a, T> {
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
                        row_data.push(format!("{}", self.get(row, col)));
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
    };
}

macro_rules! impl_matrix_read_for_row_view {
    ($view_type:ident) => {
        impl<T> MatrixRead<T> for $view_type<'_, T> {
            fn rows(&self) -> usize {
                1
            }

            fn cols(&self) -> usize {
                self.cols
            }

            fn get(&self, _row: usize, col: usize) -> &T {
                let index = self.offset + col * self.col_stride;
                &self.data[index]
            }

            fn is_row_contiguous(&self) -> bool {
                self.col_stride == 1
            }

            fn row(&self, _row: usize) -> RowView<'_, T> {
                RowView {
                    cols: self.cols,
                    data: self.data,
                    offset: self.offset,
                    col_stride: self.col_stride,
                }
            }

            fn col(&self, col: usize) -> ColView<'_, T> {
                assert!(col < self.cols, "Column index out of bounds");
                ColView {
                    rows: 1,
                    data: self.data,
                    offset: self.offset + col * self.col_stride,
                    row_stride: self.col_stride,
                }
            }
        }
    };
}

macro_rules! impl_matrix_read_for_col_view {
    ($view_type:ident) => {
        impl<T> MatrixRead<T> for $view_type<'_, T> {
            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                1
            }

            fn get(&self, row: usize, _col: usize) -> &T {
                let index = self.offset + row * self.row_stride;
                &self.data[index]
            }

            fn is_row_contiguous(&self) -> bool {
                self.row_stride == 1
            }

            fn row(&self, row: usize) -> RowView<'_, T> {
                assert!(row < self.rows, "Row index out of bounds");
                RowView {
                    cols: 1,
                    data: self.data,
                    offset: self.offset + row * self.row_stride,
                    col_stride: self.row_stride,
                }
            }

            fn col(&self, col: usize) -> ColView<'_, T> {
                assert!(col < 1, "Column index out of bounds");
                ColView {
                    rows: self.rows,
                    data: self.data,
                    offset: self.offset,
                    row_stride: self.row_stride,
                }
            }
        }
    };
}

impl_matrix_read_for_view!(MatrixView);
impl_matrix_read_for_view!(MatrixViewMut);
impl_matrix_view_fmt!(MatrixView);
impl_matrix_view_fmt!(MatrixViewMut);

impl_matrix_read_for_row_view!(RowView);
impl_matrix_read_for_row_view!(RowViewMut);
impl_matrix_read_for_col_view!(ColView);
impl_matrix_read_for_col_view!(ColViewMut);

impl<T> MatrixWrite<T> for MatrixViewMut<'_, T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let index = self.offset + row * self.row_stride + col * self.col_stride;
        &mut self.data[index]
    }

    fn row_mut(&mut self, row: usize) -> RowViewMut<'_, T> {
        assert!(row < self.rows, "Row index out of bounds");
        RowViewMut {
            cols: self.cols,
            data: self.data,
            offset: self.offset + row * self.row_stride,
            col_stride: self.col_stride,
        }
    }

    fn col_mut(&mut self, col: usize) -> ColViewMut<'_, T> {
        assert!(col < self.cols, "Column index out of bounds");
        ColViewMut {
            rows: self.rows,
            data: self.data,
            offset: self.offset + col * self.col_stride,
            row_stride: self.row_stride,
        }
    }
}

impl<T> MatrixWrite<T> for RowViewMut<'_, T> {
    fn get_mut(&mut self, _row: usize, col: usize) -> &mut T {
        let index = self.offset + col * self.col_stride;
        &mut self.data[index]
    }

    fn row_mut(&mut self, _row: usize) -> RowViewMut<'_, T> {
        RowViewMut {
            cols: self.cols,
            data: self.data,
            offset: self.offset,
            col_stride: self.col_stride,
        }
    }

    fn col_mut(&mut self, col: usize) -> ColViewMut<'_, T> {
        assert!(col < self.cols, "Column index out of bounds");
        ColViewMut {
            rows: 1,
            data: self.data,
            offset: self.offset + col * self.col_stride,
            row_stride: self.col_stride,
        }
    }
}

impl<T> MatrixWrite<T> for ColViewMut<'_, T> {
    fn get_mut(&mut self, row: usize, _col: usize) -> &mut T {
        let index = self.offset + row * self.row_stride;
        &mut self.data[index]
    }

    fn row_mut(&mut self, row: usize) -> RowViewMut<'_, T> {
        assert!(row < self.rows, "Row index out of bounds");
        RowViewMut {
            cols: 1,
            data: self.data,
            offset: self.offset + row * self.row_stride,
            col_stride: self.row_stride,
        }
    }

    fn col_mut(&mut self, _col: usize) -> ColViewMut<'_, T> {
        ColViewMut {
            rows: self.rows,
            data: self.data,
            offset: self.offset,
            row_stride: self.row_stride,
        }
    }
}
