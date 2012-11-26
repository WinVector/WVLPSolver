package com.winvector.linagl;

import java.io.Serializable;

public interface LinalgFactory<T extends Matrix<T>> extends Serializable {
	public T newMatrix(int m, int n, boolean wantSparse);
}
