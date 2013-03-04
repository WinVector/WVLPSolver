package com.winvector.linalg;

import java.io.Serializable;

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
}
