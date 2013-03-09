package com.winvector.linalg.sparse;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.BitSet;

import com.winvector.linalg.PreMatrixI;

/**
 * immutable sparse matrix
 * @author johnmount
 *
 */
public final class ColumnMatrix implements PreMatrixI {
	private static final long serialVersionUID = 1L;

	private final int rows;
	private final int cols;
	private final SparseVec[] columns;
	
	
	public ColumnMatrix(final int rows, final SparseVec[] columns) {
		this.rows = rows;
		this.cols = columns.length;
		this.columns = columns;
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
	@Override
	public double get(final int row, final int col) {
		return columns[col].get(row);
	}
	
	@Override
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
				b.append(" " + columns[j].get(i));
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
	
	

	public ColumnMatrix extractColumns(final int[] basis) {
		final int newCols = basis.length;
		final ColumnMatrix newMatrix = new ColumnMatrix(rows,newCols);
		for(int j=0;j<newCols;++j) {
			newMatrix.columns[j] = columns[basis[j]];
		}
		return newMatrix;
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
	
	@Override
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
		final int newDim = rb.length;
		final BitSet rows = new BitSet(rows());
		for(final int ri: rb) {
			rows.set(ri);
		}
		final ColumnMatrix newMat = new ColumnMatrix(newDim,cols);
		for(int j=0;j<cols;++j) {
			newMat.columns[j] = columns[j].extractRows(newDim, rows);
		}
		return newMat;
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
		final int nColIndices = x.indices.length;
		for(int jj=0;jj<nColIndices;++jj) {
			final int k = x.indices[jj];
			final double xk = x.values[jj];
			if(Math.abs(xk)>1.0e-8) {
				final SparseVec col = columns[k];
				final int nRowIndices = col.nIndices();
				for(int ii=0;ii<nRowIndices;++ii) {
					final int i = col.index(ii);
					final double vij = col.value(ii);
					r[i] += xk*vij;
				}
			}
		}
		return r;		
	}
}
