package com.winvector.linalg.sparse;

import java.util.SortedMap;



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
		final int nindices = this.indices.length;
		if(nindices>0) {
			if(indices[nindices-1]>=dim) {
				throw new IllegalArgumentException("out of bounds index");
			}
		}
	}
	
	private SparseVec(final double[] x) {
		super(x);
		dim = x.length;
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
	
	
	public static SparseVec sparseVec(final double[] x) {
		return new SparseVec(x);
	}

	
	
	SparseVec scale(final double[] scale) {
		final int nindices = indices.length;
		if(nindices<=0) {
			return this;
		}
		double[] nvalues = new double[nindices];
		for(int ii=0;ii<nindices;++ii) {
			final int i = indices[ii];
			if(null==scale) {
				nvalues[ii] = values[ii];
			} else {
				nvalues[ii] = values[ii]*scale[i];
			}
		}
		return new SparseVec(dim,indices,nvalues); // share indices
	}
	
	public SparseVec scale(final double scale) {
		final int nindices = indices.length;
		if(nindices<=0) {
			return this;
		}
		if(scale==0.0) {
			return new SparseVec(dim,new int[0],new double[0]);
		}
		double[] nvalues = new double[nindices];
		for(int ii=0;ii<nindices;++ii) {
			nvalues[ii] = values[ii]*scale;
		}
		return new SparseVec(dim,indices,nvalues); // share indices

	}
	


	public double dot(final double[] x) {
		if(x.length!=dim) {
			throw new IllegalArgumentException();
		}
		return super.dot(x);
	}
	
	public double[] toDense() {
		return toArray(dim);
	}
	
	SparseVec extractRows(final int newDim, final SortedMap<Integer,Integer> renumbering) {
		final int nindices = indices.length;
		final int nnz;
		{
			int k = 0;
			for(int ii=0;ii<nindices;++ii) {
				final int i = indices[ii];
				if(renumbering.containsKey(i)) {
					++k;
				}
			}
			nnz = k;
		}
		final int[] newIndices = new int[nnz];
		final double[] newValues = new double[nnz];
		int k = 0;
		for(int ii=0;(ii<nindices)&&(k<nnz);++ii) {
			final int i = indices[ii];
			final Integer newIndex = renumbering.get(i);
			if(null!=newIndex) {
				newIndices[k] = newIndex;
				newValues[k] = values[ii];
				++k;
			}
		}
		return new SparseVec(newDim,newIndices,newValues);
	}
	
	/**
	 * slow (discouraged)
	 * @param i
	 * @return
	 */
	@Override
	public double get(final int i) {
		if((i<0)||(i>=dim)) {
			throw new ArrayIndexOutOfBoundsException(""+i);
		}
		return super.get(i);
	}
}
