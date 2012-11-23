package com.winvector.linagl;

public interface LinalgFactory<T extends Matrix<T>> {
	public Vector newVector(int dim);
	public T newMatrix(int m, int n, boolean wantSparse);
}
