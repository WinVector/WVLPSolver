package com.winvector.linalg;

import java.io.PrintStream;
import java.io.Serializable;

import com.winvector.linalg.sparse.ColumnMatrix;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;




public abstract class Matrix<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;
	

	abstract public int cols();
	abstract public int rows();
	abstract public boolean sparseRep();
	abstract public LinalgFactory<T> factory();

	abstract public double get(int i, int j);
	abstract public void set(int i, int j, double d);

	abstract public <Z extends T> T multMat(final Z o);
	abstract public double[] solve(final double[] y);
	abstract public T inverse();
	
	abstract public Object buildExtractTemps();
	abstract public int extractColumnToTemps(int ci, Object extractTemps, int[] indices,
			double[] values);
	abstract public SparseVec extractColumn(final int ci, final Object extractTemps);

	
	public ColumnMatrix columnMatrix() {		
		final int cols = cols();
		final SparseVec[] columns = new SparseVec[cols];
		final Object extractTemps = buildExtractTemps();
		for(int j=0;j<cols;++j) {
			columns[j] = extractColumn(j,extractTemps);
		}
		return new ColumnMatrix(rows(),columns);
	}
	
	public <Z extends Matrix<Z>> Z copy(final LinalgFactory<Z> factory, final boolean wantSparse) {
		final int rows = rows();
		final int cols = cols();
		final Z r = factory.newMatrix(rows,cols,wantSparse);
		for(int i=0;i<rows;++i) {
			for(int j=0;j<cols;++j) {
				final double vij = get(i,j);
				if(vij!=0) {
					r.set(i, j, vij);
				}
			}
		}
		return r;
	}

	private double[] extractRow(final int ri) {
		final int cols = cols();
		final double[] r = new double[cols];
		for(int i=0;i<cols;++i) {
			final double e = get(ri, i);
			r[i] = e;
		}
		return r;
	}
	
	public static double[] extract(final double[] v, final int[] indices) {
		final int ilength = indices.length;
		final double[] r = new double[ilength];
		for(int i=0;i<ilength;++i) {
			r[i] = v[indices[i]];
		}
		return r;
	}



	public static double dot(final double[] x, final double[] y) {
		double r = 0.0;
		final int n = x.length;
		for(int i=0;i<n;++i) {
			r += x[i]*y[i];
		}
		return r;
	}
	
	
	public static String toString(final double[] x) {
		final StringBuilder b = new StringBuilder();
		b.append("{");
		boolean first = true;
		for(final double xi: x) {
			if(!first) {
				b.append(", ");
			} else {
				first = false;
			}
			b.append(""+xi);
		}
		b.append("}");
		return b.toString();
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append("[" + rows() + "][" + cols() + "]{\n");
		for(int i=0;i<rows();++i) {
			b.append(" ");
			b.append(toString(extractRow(i)));
			if(i<rows()-1) {
				b.append(",");
			}
			b.append("\n");
		}
		b.append("}\n");
		return b.toString();
	}
	
	public void print(final PrintStream p) {
		p.print("{");
		p.println();
		for (int i = 0; i < rows(); ++i) {
			p.print(" ");
			p.println(toString(extractRow(i)));
			p.println();
		}
		p.print("}");
		p.println();
	}

	public double[] mult(final double[] x) {
		final int cols = cols();
		if(cols!=x.length) {
			throw new IllegalArgumentException();
		}
		final int rows = rows();
		final double[] r = new double[rows];
		for(int k=0;k<cols;++k) {
			final double xk = x[k];
			if(Math.abs(xk)>1.0e-8) {
				for (int i = 0; i < rows; ++i) {
					r[i] += xk*get(i,k);
				}
			}
		}
		return r;		
	}
	
	public double[] mult(final HVec x) {
		final int rows = rows();
		final double[] r = new double[rows];
		final int nindices = x.nIndices();
		for(int ii=0;ii<nindices;++ii) {
			final int k = x.index(ii);
			final double xk = x.value(ii);
			if(Math.abs(xk)>1.0e-8) {
				for (int i = 0; i < rows; ++i) {
					r[i] += xk*get(i,k);
				}
			}
		}
		return r;		
	}

	public double[] multLeft(final double[] b) {
		final int rows = rows();
		if (rows != b.length) {
			throw new IllegalArgumentException();
		}
		final int cols = cols();
		final double[] r = new double[cols];
		for(int k=0;k<rows;++k) {
			final double bk = b[k];
			if(Math.abs(bk)>1.0e-8) {
				for (int i = 0; i < cols; ++i) {
					r[i] += bk*get(k, i);
				}
			}
		}
		return r;
	}
	
	
	public void setRow(final int ri, final double[] row) {
		final int cols = cols();
		for(int i=0;i<cols;++i) {
			final double e = row[i];
			set(ri,i, e);
		}
	}
}
