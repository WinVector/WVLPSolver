package com.winvector.linalg;

import java.io.Serializable;

import com.winvector.linalg.sparse.SparseVec;

public abstract class LinalgFactory<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;

	abstract public T newMatrix(int m, int n, boolean wantSparse);
	
	public T identityMatrix(final int m, boolean wantSparse) {
		final T r = newMatrix(m,m,wantSparse);
		for(int i=0;i<m;++i) {
			r.set(i, i, 1.0);
		}
		return r;
	}
	
	public T matrixCopy(final PreMatrixI a) {
		final int cols = a.cols();
		final int rows = a.rows();
		final Object extractTemps = a.buildExtractTemps();
		int npop = 0;
		for(int j=0;j<cols;++j) {
			final SparseVec col = a.extractColumn(j, extractTemps);
			npop += col.popCount();
		}
		boolean wantSparse = npop<(0.1*rows)*cols;		
		final T m = newMatrix(rows,cols,wantSparse);
		for(int j=0;j<cols;++j) {
			final SparseVec col = a.extractColumn(j, extractTemps);
			final int colindiceslength = col.nIndices();
			for(int ii=0;ii<colindiceslength;++ii) {
				final double aij = col.value(ii);
				if(0.0!=aij) {
					m.set(col.index(ii),j,aij);
				}
			}
		}
		return m;
	}
}
