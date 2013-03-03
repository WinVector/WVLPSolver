package com.winvector.linagl;


/**
 * immutable vector, pop count can not be taking from length (some zero entries may be stored)
 * @author johnmount
 *
 */
public final class SparseVec extends HVec {
	private static final long serialVersionUID = 1L;
	
	public final int dim;
		
	public SparseVec(final int dim, final int[] indices, final double[] values) {
		super(indices,values);
		this.dim = dim;
		final int nindices = indices.length;
		if(nindices>0) {
			if(indices[nindices-1]>=dim) {
				throw new IllegalArgumentException("out of bounds index");
			}
		}
	}
	
	public static SparseVec sparseVec(final int dim, final int coord, final double val) {
		if(val==0) {
			return new SparseVec(dim,new int[0],new double[0]);
		} else {
			if((coord<0)||(coord>dim)) {
				throw new IllegalArgumentException();
			}
			return new SparseVec(dim,new int[] { coord }, new double[] {val});
		}
	}
	
	
	static SparseVec sparseVec(final double[] x) {
		final int dim = x.length;
		int index = 0;
		for(int i=0;i<dim;++i) {
			if(x[i]!=0) {
				++index;
			}
		}
		final int nnz = index;
		final int[] indices = new int[nnz];
		final double[] values = new double[nnz];
		index = 0;
		for(int i=0;(i<dim)&&(index<nnz);++i) {
			if(x[i]!=0) {
				indices[index] = i;
				values[index] = x[i];
				++index;
			}
		}
		return new SparseVec(dim,indices,values);
	}
	
	
	SparseVec scale(final double[] scale) {
		SparseVec v = new SparseVec(dim,indices,new double[values.length]); // share indices
		final int nindices = indices.length;
		for(int ii=0;ii<nindices;++ii) {
			final int i = indices[ii];
			if(null==scale) {
				v.values[ii] = values[ii];
			} else {
				v.values[ii] = values[ii]*scale[i];
			}
		}
		return v;
	}
	


	public double dot(final double[] x) {
		if(x.length!=dim) {
			throw new IllegalArgumentException();
		}
		return super.dot(x);
	}
	
	

	public double[] toDense() {
		final double[] x = new double[dim];
		final int nindices = indices.length;
		for(int ii=0;ii<nindices;++ii) {
			final double vi = values[ii];
			if(0.0!=vi) {
				x[indices[ii]] = vi;
			}
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
		return super.get(i);
	}
}
