package com.winvector.linalg.colt;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;



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
	public double[] solve(final double[] bIn) {
		if(bIn.length!=rows()) {
			throw new IllegalArgumentException();
		}
		final DoubleMatrix2D b = new DenseDoubleMatrix2D(rows(),1);
		for(int i=0;i<rows();++i) {
			b.set(i,0,bIn[i]);
		}
		final DoubleMatrix2D a = underlying;
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

	private static final class ExtractTemps {
		public final IntArrayList indexList;
		public final DoubleArrayList valueList;
		
		public ExtractTemps(final int m) {
			indexList = new IntArrayList(m);
			valueList = new DoubleArrayList(m);
		}
	}
	
	@Override
	public Object buildExtractTemps() {
		return new ExtractTemps(rows());
	}

	@Override
	public SparseVec extractColumn(final int ci, final Object extractTemps) {
		final ExtractTemps et = (ExtractTemps)extractTemps;
		final DoubleMatrix1D col = underlying.viewColumn(ci);
		col.getNonZeros(et.indexList,et.valueList);
		final int k = et.indexList.size();
		final int[] indices = new int[k];
		final double[] values = new double[k];
		for(int ii=0;ii<k;++ii) {
			final int index = et.indexList.get(ii);
			final double value = et.valueList.get(ii);
			indices[ii] = index;
			values[ii] = value;
		}
		return new SparseVec(rows(),indices,values);
	}
	

	
	@Override
	public double[] mult(final double[] x) {
		return Algebra.ZERO.mult(underlying,new DenseDoubleMatrix1D(x)).toArray();
	}
	
	@Override
	public double[] mult(final HVec x) {
		return Algebra.ZERO.mult(underlying,new DenseDoubleMatrix1D(x.toArray(underlying.columns()))).toArray();
	}
	
	@Override
	public double[] multLeft(final double[] b) {
		return Algebra.ZERO.mult(Algebra.ZERO.transpose(underlying),new DenseDoubleMatrix1D(b)).toArray();
	}
}
