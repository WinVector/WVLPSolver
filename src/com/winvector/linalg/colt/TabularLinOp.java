package com.winvector.linalg.colt;

import com.winvector.linagl.LinOpI;
import com.winvector.linagl.PreMatrixI;

public final class TabularLinOp implements LinOpI {
	private final int rows;
	private final int cols;
	private final int[] iv;
	private final int[] jv;
	private final double[] av;
	private int k;
	
	public TabularLinOp(final int rows, final int cols) {
		this.rows = rows;
		this.cols = cols;
		final int size = rows*cols;
		iv = new int[size];
		jv = new int[size];
		av = new double[size];
		k = 0;
	}

	@Override
	public int rows() {
		return rows;
	}

	@Override
	public int cols() {
		return cols;
	}

	public void setV(final PreMatrixI m) {
		if((m.rows()!=rows)||(m.cols()!=cols)) {
			throw new IllegalArgumentException();
		}
		k = 0;
		for(int j=0;j<cols;++j) {
			for(int i=0;i<rows;++i) {
				final double aij = m.get(i, j); 
				if(Math.abs(aij)>1.0e-20) {
					iv[k] = i;
					jv[k] = j;
					av[k] = aij;
					++k;
				}
			}
		}
	}
	
	@Override
	public double[] multLeft(final double[] y) {
		if(rows!=y.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[cols];
		for(int ii=0;ii<k;++ii) {
			final int i = iv[ii];
			final int j = jv[ii];
			final double aij = av[ii];
			final double yi = y[i];
			if(yi!=0.0) {
				r[j] += aij*yi;
			}
		}
		return r;
	}

	@Override
	public double[] mult(final double[] x) {
		if(cols!=x.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[rows];
		for(int ii=0;ii<k;++ii) {
			final int i = iv[ii];
			final int j = jv[ii];
			final double aij = av[ii];
			final double xj = x[j];
			if(xj!=0.0) {
				r[i] += aij*xj;
			}
		}
		return r;
	}

}
