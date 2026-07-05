use super::matrix::MatrixRead;
use std::fmt;

pub struct MatrixView<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

pub struct MatrixViewMut<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a mut [T],
    pub(crate) offset: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

pub struct RowView<'a, T> {
    pub cols: usize,
    pub data: &'a [T],
    pub offset: usize,
    pub col_stride: usize,
}

pub struct RowViewMut<'a, T> {
    pub cols: usize,
    pub data: &'a mut [T],
    pub offset: usize,
    pub col_stride: usize,
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

impl_matrix_read_for_view!(MatrixView);
impl_matrix_read_for_view!(MatrixViewMut);
impl_matrix_view_fmt!(MatrixView);
impl_matrix_view_fmt!(MatrixViewMut);

impl<T> MatrixRead<T> for RowView<'_, T> {
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
}

impl<T> MatrixRead<T> for RowViewMut<'_, T> {
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
}

impl<T> MatrixRead<T> for ColView<'_, T> {
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
}
