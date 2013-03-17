package com.winvector.linalg.jblas;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;



public class JBlasMatrix extends Matrix<JBlasMatrix> {
	private static final long serialVersionUID = 1L;
	
	private final DoubleMatrix underlying;
	
	private JBlasMatrix(final int m, final int n, final boolean wantSparse) {
		underlying = new DoubleMatrix(m,n);
	}
	
	private JBlasMatrix(final DoubleMatrix o, final boolean wantSparse) {
		this.underlying = o;
	}

	@Override
	public int cols() {
		return underlying.getColumns();
	}

	@Override
	public int rows() {
		return underlying.getRows();
	}

	@Override
	public double get(final int row, final int col) {
		return underlying.get(row,col);
	}

	@Override
	public void set(final int row, final int col, final double v) {
		underlying.put(row,col,v);
	}
	
	@Override
	public double[] solve(final double[] bIn) {
		if(bIn.length!=rows()) {
			throw new IllegalArgumentException();
		}
		final DoubleMatrix b = new DoubleMatrix(rows(),1);
		for(int i=0;i<rows();++i) {
			b.put(i,0,bIn[i]);
		}
		final DoubleMatrix p = Solve.solve(underlying, b);
		final double[] r = new double[cols()];
		for(int i=0;i<r.length;++i) {
			r[i] = p.get(i,0);
		}
		return r;
	}
	
	@Override
	public String toString() {
		return underlying.toString();
	}

	@Override
	public JBlasMatrix inverse() {
		final int n = rows();
		final DoubleMatrix b = new DoubleMatrix(n,n);
		for(int i=0;i<n;++i) {
			b.put(i,i,1.0);
		}
		final DoubleMatrix p = Solve.solve(underlying, b);
		return new JBlasMatrix(p,false);
	}
	
	@Override
	public <Z extends JBlasMatrix> JBlasMatrix multMat(final Z o) {
		if(cols()!=o.rows()) {
			throw new IllegalArgumentException();
		}
		return new JBlasMatrix(underlying.mmul(o.underlying),false);
	}

	@Override
	public boolean sparseRep() {
		return false;
	}
	
	public static final LinalgFactory<JBlasMatrix> factory = new LinalgFactory<JBlasMatrix>() {
		private static final long serialVersionUID = 1L;

		@Override
		public JBlasMatrix newMatrix(int m, int n, boolean wantSparse) {
			return new JBlasMatrix(m,n,wantSparse);
		}
	};
	
	@Override
	public LinalgFactory<JBlasMatrix> factory() {
		return factory;
	}

	
	@Override
	public Object buildExtractTemps() {
		return null;
	}

	@Override
	public int extractColumnToTemps(final int ci, final Object extractTemps,
			final int[] indices, final double[] values) {
		final int rows = rows();
		int k = 0;
		for(int i=0;i<rows;++i) {
			final double e = get(i, ci);
			if(e!=0.0) {
				values[k] = e;
				indices[k] = i;
				++k;
			}
		}
		return k;
	}
	
	@Override
	public SparseVec extractColumn(final int ci, final Object extractTemps) {
		final int rows = rows();
		int k = 0;
		for(int i=0;i<rows;++i) {
			final double e = get(i, ci);
			if(e!=0.0) {
				++k;
			}
		}
		final int nnz = k;
		k = 0;
		final int[] indices = new int[nnz];
		final double[] values = new double[nnz];
		for(int i=0;(i<rows)&&(k<nnz);++i) {
			final double e = get(i, ci);
			if(e!=0.0) {
				values[k] = e;
				indices[k] = i;
				++k;
			}
		}
		return new SparseVec(rows,indices,values);
	}

	
	@Override
	public double[] mult(final double[] x) {
		final DoubleMatrix o = new DoubleMatrix(x);
		return underlying.mmul(o).toArray();
	}
	
	@Override
	public double[] mult(final HVec x) {
		return mult(x.toArray(cols()));
	}
	
	@Override
	public double[] multLeft(final double[] b) {
		final DoubleMatrix o = new DoubleMatrix(1,b.length);
		for(int i=0;i<b.length;++i) 
			o.put(0,i,b[i]);
		return o.mmul(underlying).toArray();
	}
}
