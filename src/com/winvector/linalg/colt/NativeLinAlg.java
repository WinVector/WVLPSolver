package com.winvector.linalg.colt;

import com.winvector.linagl.LinalgFactory;

public final class NativeLinAlg implements LinalgFactory<NativeMatrix> {
	private NativeLinAlg() {
	}
	
	public static final LinalgFactory<NativeMatrix> factory = new NativeLinAlg();

	@Override
	public NativeVector newVector(final int n) {
		return new NativeVector(n);
	}
	
	@Override
	public NativeMatrix newMatrix(final int m, final int n, final boolean wantSparse) {
		return new NativeMatrix(m,n,wantSparse);
	}
}
