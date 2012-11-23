package com.winvector.linalg.colt;

import com.winvector.linagl.LinalgFactory;

public final class ColtLinAlg implements LinalgFactory<ColtMatrix> {
	private ColtLinAlg() {
	}
	
	public static final LinalgFactory<ColtMatrix> factory = new ColtLinAlg();

	@Override
	public NativeVector newVector(final int n) {
		return new NativeVector(n);
	}
	
	@Override
	public ColtMatrix newMatrix(final int m, final int n, final boolean wantSparse) {
		return new ColtMatrix(m,n,wantSparse);
	}
}
