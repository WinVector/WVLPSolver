package com.winvector.linalg;

import java.io.PrintStream;
import java.util.ArrayList;

import com.winvector.linalg.colt.NativeMatrix;

/**
 * immutable sparse matrix
 * @author johnmount
 *
 */
public final class ColumnMatrix implements PreMatrixI {
	private static final long serialVersionUID = 1L;

	public final int rows;
	public final int cols;
	private final SparseVec[] columns;
	
	public ColumnMatrix(final PreMatrixI a) {
		rows = a.rows();
		cols = a.cols();
		columns = new SparseVec[cols];
		final Object extractTemps = a.buildExtractTemps();
		for(int j=0;j<cols;++j) {
			columns[j] = a.extractColumn(j,extractTemps);
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

	@Override
	public Object buildExtractTemps() {
		return null;
	}
	
	@Override
	public SparseVec extractColumn(final int j, final Object extractTemps) {
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
			if(x[j]!=0) {
				final SparseVec col = columns[j];
				final int colindiceslength = col.indices.length;
				for(int ii=0;ii<colindiceslength;++ii) {
					final int i = col.indices[ii];
					final double aij = col.values[ii];
					if(0.0!=aij) {
						res[i] += aij*x[j];
					}
				}
			}
		}
		return res;
	}



	public double[] multLeft(final double[] y) {
		final double[] res = new double[cols];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final int i = col.indices[ii];
				final double aij = col.values[ii];
				if(0.0!=aij) {
					res[j] += aij*y[i];
				}
			}
		}
		return res;
	}
	
	

	public <T extends Matrix<T>> T extractColumns(final int[] basis,
			final LinalgFactory<T> factory) {
		int npop = 0;
		final int blength = basis.length;
		for(int jj=0;jj<blength;++jj) {
			final int j = basis[jj];
			npop += columns[j].popCount();
		}
		boolean wantSparse = npop<(0.1*rows)*blength;
		final T r = factory.newMatrix(rows,blength,wantSparse);
		for(int jj=0;jj<blength;++jj) {
			final int j = basis[jj];
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final double aij = col.values[ii];
				if(0.0!=aij) {
					r.set(col.indices[ii],jj,aij);
				}
			}
		}
		return r;
	}

	
	public <T extends Matrix<T>> T matrixCopy(final LinalgFactory<T> factory) {
		int npop = 0;
		for(int j=0;j<cols;++j) {
			npop += columns[j].popCount();
		}
		boolean wantSparse = npop<(0.1*rows)*cols;		
		final T m = factory.newMatrix(rows,cols,wantSparse);
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final double aij = col.values[ii];
				if(0.0!=aij) {
					m.set(col.indices[ii],j,aij);
				}
			}
		}
		return m;
	}

	public ColumnMatrix addColumns(final ArrayList<SparseVec> cs) {
		final int cssize = cs.size();
		final ColumnMatrix r = new ColumnMatrix(rows,cols+cssize);
		for(int i=0;i<cols;++i) {
			r.columns[i] = columns[i];
		}
		for(int i=0;i<cssize;++i) {
			if(cs.get(i).dim!=rows) {
				throw new IllegalArgumentException();
			}
			r.columns[cols+i] = cs.get(i);
		}
		return r;
	}
	
	public double[] sumAbsRowValues() {
		final double[] r = new double[rows];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final double aij = col.values[ii];
				if(0.0!=aij) {
					final int i = col.indices[ii];
					r[i] += Math.abs(aij);
				}
			}
		}
		return r;
	}
	
	public ColumnMatrix extractRows(final int[] rb) {
		// TODO better implementation
		return new ColumnMatrix(matrixCopy(NativeMatrix.factory).extractRows(rb,NativeMatrix.factory));
	}

	public ColumnMatrix rescaleRows(double[] scale) {
		final ColumnMatrix r = new ColumnMatrix(rows,cols);
		for(int j=0;j<cols;++j) {
			r.columns[j] = columns[j].scale(scale);
		}
		return r;
	}

	@Override
	public int rows() {
		return rows;
	}

	@Override
	public int cols() {
		return cols;
	}

	@Override
	public double[] mult(final HVec x) {
		final int rows = rows();
		final double[] r = new double[rows];
		final int nindices = x.indices.length;
		for(int ii=0;ii<nindices;++ii) {
			final int k = x.indices[ii];
			final double xk = x.values[ii];
			if(Math.abs(xk)>1.0e-8) {
				for (int i = 0; i < rows; ++i) {
					r[i] += xk*get(i,k);
				}
			}
		}
		return r;		
	}
}
