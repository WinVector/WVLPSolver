package com.winvector.sparse;

import java.io.Serializable;
import java.util.Arrays;

import com.winvector.linagl.Matrix;

/**
 * immutable vector
 * @author johnmount
 *
 */
public final class SparseVec implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final int dim;
	final int[] indices;
	final double[] values;
	
	public SparseVec(final double[] x) {
		dim = x.length;
		int index = 0;
		for(int i=0;i<dim;++i) {
			if(x[i]!=0) {
				++index;
			}
		}
		final int nnz = index;
		indices = new int[nnz];
		values = new double[nnz];
		index = 0;
		for(int i=0;(i<dim)&&(index<nnz);++i) {
			if(x[i]!=0) {
				indices[index] = i;
				values[index] = x[i];
				++index;
			}
		}
	}
	
	public double dot(final double[] x) {
		if(x.length!=dim) {
			throw new IllegalArgumentException();
		}
		double d = 0.0;
		for(int ii=0;ii<indices.length;++ii) {
			d += values[ii]*x[indices[ii]];
		}
		return d;
	}
	
	public static <T extends Matrix<T>> double[] mult(final T m, final SparseVec x) {
		if(m.cols()!=x.dim) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[m.rows()];
		for (int i = 0; i < r.length; ++i) {
			double z = 0;
			for(int kk=0;kk<x.indices.length;++kk) {
				z += x.values[kk]*m.get(i,x.indices[kk]);
			}
			r[i] = z;
		}
		return r;		
	}
	
	/**
	 * slow (discouraged)
	 * @param i
	 * @return
	 */
	double get(final int i) {
		if((i<0)||(i>=dim)) {
			throw new ArrayIndexOutOfBoundsException(""+i);
		}
		final int ii = Arrays.binarySearch(indices,i);
		if((ii<0)||(ii>=indices.length)) {
			return 0.0;
		}
		if(i!=indices[ii]) {
			throw new IllegalStateException("lookup failed " + i + " -> " + ii + "\t" + indices[ii]);
		}
		return values[ii];
	}
}
