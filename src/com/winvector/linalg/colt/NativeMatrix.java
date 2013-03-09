package com.winvector.linalg.colt;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.SparseVec;


public class NativeMatrix extends Matrix<NativeMatrix> {
	private static final long serialVersionUID = 1L;
	
	private final int rows;
	private final int cols;
	private final double[][] u;
	
	private NativeMatrix(final int m, final int n, final boolean wantSparse) {
		this.rows = m;
		this.cols = n;
		u = new double[m][n];
	}
	
	public NativeMatrix(final double[][] u) {
		this.u = u;
		this.rows = u.length;
		this.cols = u.length;
	}

	@Override
	public int cols() {
		return cols;
	}

	@Override
	public int rows() {
		return rows;
	}

	@Override
	public double get(final int row, final int col) {
		return u[row][col];
	}

	@Override
	public void set(final int row, final int col, final double v) {
		u[row][col] = v;
	}

	@Override
	public double[] solve(final double[] bIn) {
		if(bIn.length!=rows()) {
			throw new IllegalArgumentException();
		}
		final DoubleMatrix2D b = new DenseDoubleMatrix2D(rows(),1);
		for(int i=0;i<rows();++i) {
			b.set(i,0,bIn[i]);
		}
		final DoubleMatrix2D a = new DenseDoubleMatrix2D(u);
		final DoubleMatrix2D p = Algebra.ZERO.solve(a,b);
		final double[] r = new double[cols()];
		for(int i=0;i<r.length;++i) {
			r[i] = p.get(i,0);
		}
		return r;
	}

	
	@Override
	public NativeMatrix inverse() {
		if(rows!=cols) {
			throw new IllegalArgumentException();
		}
		final DoubleMatrix2D ainv = Algebra.ZERO.inverse(new DenseDoubleMatrix2D(u));
		final NativeMatrix r = new NativeMatrix(rows,cols,false);
		for(int i=0;i<rows;++i) {
			for(int j=0;j<cols;++j) {
				r.u[i][j] = ainv.get(i, j);
			}
		}
		return r;
	}
	

	@Override
	public Object buildExtractTemps() {
		return null;
	}
	
	@Override
	public SparseVec extractColumn(final int ci, final Object extractTemps) {
		final int rows = rows();
		final double[] r = new double[rows];
		for(int i=0;i<rows;++i) {
			final double e = get(i, ci);
			r[i] = e;
		}
		return SparseVec.sparseVec(r);
	}
	
	@Override
	public <Z extends NativeMatrix> NativeMatrix multMat(final Z o) {
		if(cols!=o.rows) {
			throw new IllegalArgumentException();
		}
		final NativeMatrix r = new NativeMatrix(rows,o.cols,false);
		for(int i=0;i<rows;++i) {
			for(int j=0;j<o.cols;++j) {
				double v = 0.0;
				for(int k=0;k<cols;++k) {
					v += u[i][k]*o.u[k][j];
				}
				r.u[i][j] = v;
			}
		}
		return r;
	}

	@Override
	public boolean sparseRep() {
		return false;
	}

	public static final LinalgFactory<NativeMatrix> factory = new LinalgFactory<NativeMatrix>() {
		private static final long serialVersionUID = 1L;

		@Override
		public NativeMatrix newMatrix(int m, int n, boolean wantSparse) {
			return new NativeMatrix(m,n,wantSparse);
		}
	};
	
	@Override
	public LinalgFactory<NativeMatrix> factory() {
		return factory;
	}
}

