package com.winvector.lp;

import java.io.Serializable;
import java.util.Arrays;

/**
 * immutable vector
 * @author johnmount
 *
 */
public final class SparseVec implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final int dim;
	public final int[] indices;
	public final double[] values;
	
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
	
	private SparseVec(final int dim, final int pop) {
		this.dim = dim;
		indices = new int[pop];
		values = new double[pop];
	}
	
	public SparseVec copy() {
		final SparseVec v = new SparseVec(dim,indices.length);
		for(int ii=0;ii<indices.length;++ii) {
			v.indices[ii] = indices[ii];
			v.values[ii] = values[ii];
		}
		return v;
	}
	
	public double[] denseCopy() {
		final double[] x = new double[dim];
		for(int ii=0;ii<indices.length;++ii) {
			x[ii] = values[ii];
		}
		return x;
	}
	
	/**
	 * slow (discouraged)
	 * @param i
	 * @return
	 */
	public double get(final int i) {
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
