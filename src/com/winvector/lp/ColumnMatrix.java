package com.winvector.lp;

import java.io.PrintStream;
import java.io.Serializable;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.ColtMatrix;

public final class ColumnMatrix implements Serializable {
	private static final long serialVersionUID = 1L;

	public final int rows;
	public final int cols;
	private final SparseVec[] columns;
	
	public <T extends Matrix<T>> ColumnMatrix(final Matrix<T> a) {
		rows = a.rows();
		cols = a.cols();
		columns = new SparseVec[cols];
		for(int j=0;j<cols;++j) {
			columns[j] = new SparseVec(a.extractColumn(j));
		}
	}

	private ColumnMatrix(final int rows, final int cols) {
		this.rows = rows;
		this.cols = cols;
		columns = new SparseVec[cols];
	}
	
	/**
	 * slow (discouraged)
	 * @param row
	 * @param col
	 * @return
	 */
	public double get(final int row, final int col) {
		return columns[col].get(row);
	}


	public SparseVec extractColumn(final int j) {
		return columns[j];
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append("[" + rows + "][" + cols + "]{\n");
		for(int i=0;i<rows;++i) {
			b.append(" {");
			for(int j=0;j<cols;++j) {
				if(j>0) {
					b.append(",");
				}
				b.append(" " + get(i,j));
			}
			b.append("}");
			if(i<rows-1) {
				b.append(",");
			}
			b.append("\n");
		}
		b.append("}\n");
		return b.toString();
	}

	public void print(final PrintStream p) {
		p.println(toString());
	}



	public double[] mult(final double[] x) {
		final double[] res = new double[rows];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			for(int ii=0;ii<col.indices.length;++ii) {
				final int i = col.indices[ii];
				final double aij = col.values[ii];
				res[i] += aij*x[j];
			}
		}
		return res;
	}



	public double[] multLeft(final double[] y) {
		final double[] res = new double[cols];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			for(int ii=0;ii<col.indices.length;++ii) {
				final int i = col.indices[ii];
				final double aij = col.values[ii];
				res[j] += aij*y[i];
			}
		}
		return res;
	}
	
	

	public <T extends Matrix<T>> T extractColumns(final int[] basis,
			final LinalgFactory<T> factory) {
		int npop = 0;
		for(int jj=0;jj<basis.length;++jj) {
			final int j = basis[jj];
			npop += columns[j].indices.length;
		}
		boolean wantSparse = npop<(0.1*rows)*basis.length;
		final T r = factory.newMatrix(rows,basis.length,wantSparse);
		for(int jj=0;jj<basis.length;++jj) {
			final int j = basis[jj];
			final SparseVec col = columns[j];
			for(int ii=0;ii<col.indices.length;++ii) {
				r.set(col.indices[ii],jj,col.values[ii]);
			}
		}
		return r;
	}

	
	public ColtMatrix matrixCopy() {
		int npop = 0;
		for(int j=0;j<cols;++j) {
			npop += columns[j].indices.length;
		}
		boolean wantSparse = npop<(0.1*rows)*cols;		
		final ColtMatrix m = new ColtMatrix(rows,cols,wantSparse);
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			for(int ii=0;ii<col.indices.length;++ii) {
				m.set(col.indices[ii],j,col.values[ii]);
			}
		}
		return m;
	}

	public ColumnMatrix extractRows(final int[] rb) {
		// TODO better implementation
		return new ColumnMatrix(matrixCopy().extractRows(rb,ColtMatrix.factory));
	}

	public ColumnMatrix addColumn(final double[] b) {
		if(b.length!=rows) {
			throw new IllegalArgumentException();
		}
		final ColumnMatrix r = new ColumnMatrix(rows,cols+1);
		for(int i=0;i<cols;++i) {
			r.columns[i] = columns[i];
		}
		r.columns[cols] = new SparseVec(b);
		return r;
	}
}
