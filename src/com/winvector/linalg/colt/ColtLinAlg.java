package com.winvector.linalg.colt;

import com.winvector.linagl.LinalgFactory;

public final class ColtLinAlg implements LinalgFactory<ColtMatrix> {
	private static final long serialVersionUID = 1L;
	

	private ColtLinAlg() {
	}
	
	public static final LinalgFactory<ColtMatrix> factory = new ColtLinAlg();
	
	@Override
	public ColtMatrix newMatrix(final int m, final int n, final boolean wantSparse) {
		return new ColtMatrix(m,n,wantSparse);
	}
}
