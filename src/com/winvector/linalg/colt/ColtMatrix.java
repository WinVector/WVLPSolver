package com.winvector.linalg.colt;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;



public class ColtMatrix extends Matrix<ColtMatrix> {
	private static final long serialVersionUID = 1L;
	
	private final boolean wantSparse;
	private final DoubleMatrix2D underlying;
	
	private ColtMatrix(final int m, final int n, final boolean wantSparse) {
		this.wantSparse = wantSparse;
		if(wantSparse) {
			underlying = new SparseDoubleMatrix2D(m,n);
		} else {
			underlying = new DenseDoubleMatrix2D(m,n);
		}
	}
	
	private ColtMatrix(final DoubleMatrix2D o, final boolean wantSparse) {
		this.wantSparse = wantSparse;
		this.underlying = o;
	}

	@Override
	public int cols() {
		return underlying.columns();
	}

	@Override
	public int rows() {
		return underlying.rows();
	}

	@Override
	public double get(final int row, final int col) {
		return underlying.get(row,col);
	}

	@Override
	public void set(final int row, final int col, final double v) {
		underlying.set(row,col,v);
	}
	
	@Override
	public double[] solve(final double[] bIn, final boolean leastsq) {
		if(bIn.length!=rows()) {
			throw new IllegalArgumentException();
		}
		DoubleMatrix2D b = new DenseDoubleMatrix2D(rows(),1);
		for(int i=0;i<rows();++i) {
			b.set(i,0,bIn[i]);
		}
		DoubleMatrix2D a = underlying;
		if(leastsq) {
			final DoubleMatrix2D at = Algebra.ZERO.transpose(a);
			a = Algebra.ZERO.mult(at,a);
			b = Algebra.ZERO.mult(at,b);
		}
		final DoubleMatrix2D p = Algebra.ZERO.solve(a,b);
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
	public ColtMatrix inverse() {
		return new ColtMatrix(Algebra.ZERO.inverse(underlying),wantSparse);
	}
	
	@Override
	public <Z extends ColtMatrix> ColtMatrix multMat(final Z o) {
		if(cols()!=o.rows()) {
			throw new IllegalArgumentException();
		}
		return new ColtMatrix(Algebra.ZERO.mult(underlying,o.underlying),wantSparse);
	}

	@Override
	public boolean sparseRep() {
		return wantSparse;
	}
	
	public static final LinalgFactory<ColtMatrix> factory = new LinalgFactory<ColtMatrix>() {
		private static final long serialVersionUID = 1L;

		@Override
		public ColtMatrix newMatrix(int m, int n, boolean wantSparse) {
			return new ColtMatrix(m,n,wantSparse);
		}
	};
	
	@Override
	public LinalgFactory<ColtMatrix> factory() {
		return factory;
	}
}
