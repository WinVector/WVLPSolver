package com.winvector.linalg;

import java.util.Arrays;

import com.winvector.linalg.sparse.HVec;


public final class TabularLinOp implements LinOpI {
	private final double epsilon = 1.0e-10;
	private final int size;
	private final int rows;
	private final int cols;
	private final int[] iv;
	private final int[] jv;
	private final double[] av;
	private int k;
	private final int[] nextWithSameI;
	private final int[] nextWithSameJ;
	private final int[] firstWithI;
	private final int[] firstWithJ;
	private boolean valid = false;
	
	public TabularLinOp(final int rows, final int cols, final int maxCells) {
		this.rows = rows;
		this.cols = cols;
		size = Math.min(maxCells,(int)Math.min(1L<<30,((long)rows)*((long)cols)));
		iv = new int[size];
		jv = new int[size];
		av = new double[size];
		nextWithSameI = new int[size];
		nextWithSameJ = new int[size];
		firstWithI = new int[rows];
		firstWithJ = new int[cols];
		Arrays.fill(firstWithI,-1);
		Arrays.fill(firstWithJ,-1);
		k = 0;
	}
	
	public boolean valid() {
		return valid;
	}
	
	public void invalidate() {
		valid = false;
	}

	@Override
	public int rows() {
		return rows;
	}

	@Override
	public int cols() {
		return cols;
	}

	public <T extends Matrix<T>> void setV(final T m) {
		if((m.rows()!=rows)||(m.cols()!=cols)) {
			throw new IllegalArgumentException();
		}
		k = 0;
		valid = false;
		Arrays.fill(firstWithI,-1);
		Arrays.fill(firstWithJ,-1);
		final int[] lastSameI = new int[rows];
		final int[] lastSameJ = new int[cols];
		Arrays.fill(lastSameI,-1);
		Arrays.fill(lastSameJ,-1);
		final Object extractTemps = m.buildExtractTemps();
		final int[] tmpIndices = new int[rows];
		final double[] tmpValues = new double[rows];
		for(int j=0;j<cols;++j) {
			final int nRowIndices = m.extractColumnToTemps(j, extractTemps, tmpIndices, tmpValues);
			for(int ii=0;ii<nRowIndices;++ii) {
				final int i = tmpIndices[ii];
				final double mij = tmpValues[ii];
				if(Math.abs(mij)>epsilon) {
					if(k>=size) {
						valid = false;
						return;
					}
					iv[k] = i;
					jv[k] = j;
					av[k] = mij;
					nextWithSameI[k] = -1;
					nextWithSameJ[k] = -1;
					if(firstWithI[i]<0) {
						firstWithI[i] = k;
					}
					if(firstWithJ[j]<0) {
						firstWithJ[j] = k;
					}
					if(lastSameI[i]>=0) {
						nextWithSameI[lastSameI[i]] = k;
					}
					if(lastSameJ[j]>=0) {
						nextWithSameJ[lastSameJ[j]] = k;
					}
					// prepare for next pass
					lastSameI[i] = k;
					lastSameJ[j] = k;
					++k;
				}
			}
		}
		valid = true;
	}
	
	@Override
	public double[] multLeft(final double[] y) {
		if(!valid) {
			throw new IllegalArgumentException();
		}
		if(rows!=y.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[cols];
		for(int i=0;i<rows;++i) {
			final double yi = y[i];
			if(Math.abs(yi)>epsilon) {
				int ii = firstWithI[i];
				while(ii>=0) {
					final int j = jv[ii];
					final double aij = av[ii];
					r[j] += aij*yi;
					ii = nextWithSameI[ii];
				}
			}			
		}
		return r;
	}

	@Override
	public double[] mult(final double[] x) {
		if(!valid) {
			throw new IllegalArgumentException();
		}
		if(cols!=x.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[rows];
		for(int j=0;j<cols;++j) {
			final double xj = x[j];
			if(Math.abs(xj)>epsilon) {
				int ii = firstWithJ[j];
				while(ii>=0) {
					final int i = iv[ii];
					final double aij = av[ii];
					r[i] += aij*xj;
					ii = nextWithSameJ[ii];
				}
			}
		}
		return r;
	}

	@Override
	public double[] mult(final HVec x) {
		if(!valid) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[rows];
		final int nindices = x.nIndices();
		for(int jj=0;jj<nindices;++jj) {
			final int j = x.index(jj);
			final double xj = x.value(jj);
			if(Math.abs(xj)>epsilon) {
				int ii = firstWithJ[j];
				while(ii>=0) {
					final int i = iv[ii];
					final double aij = av[ii];
					r[i] += aij*xj;
					ii = nextWithSameJ[ii];
				}
			}
		}
		return r;
	}
}
