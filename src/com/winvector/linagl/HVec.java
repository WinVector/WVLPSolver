package com.winvector.linagl;

import java.io.Serializable;
import java.util.Arrays;


/**
 * finite set of possible non-zero entries of a possibly infinite vector
 * @author johnmount
 *
 */
public class HVec implements Serializable {
	private static final long serialVersionUID = 1L;
	
	final int[] indices;    // do not alter!
	final double[] values;  // do not alter!
	
	public HVec(final int[] indices, final double[] values) {
		this.indices = indices;
		this.values = values;
		if(indices.length!=values.length) {
			throw new IllegalArgumentException("indices.length=" + indices.length 
					+ ", values.length=" + values.length);
		}
		final int nindices = indices.length;
		if(nindices>0) {
			if(indices[0]<0) {
				throw new IllegalArgumentException("negative index");
			}
			for(int i=0;i<nindices-1;++i) {
				if(indices[i+1]<=indices[i]) {
					throw new IllegalArgumentException("disordered indices");
				}
			}
		}
	}
	
	public int nIndices() {
		return indices.length;
	}
	
	public int index(final int ii) {
		return indices[ii];
	}
	
	public double value(final int ii) {
		return values[ii];
	}
	
	/**
	 * build from dense vector (not the prefered construciton)
	 * @param x
	 * @return
	 */
	public static HVec hVec(final double[] x) {
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
		return new HVec(indices,values);
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append("[");
		final int nindices = indices.length;
		for(int ii=0;ii<nindices;++ii) {
			if(ii>0) {
				b.append("\t");
			}
			b.append("" + indices[ii] + ":" + values[ii]);
		}
		b.append("]");
		return b.toString();
	}
	
	public int popCount() {
		int npop = 0;
		for(final double vi: values) {
			if(0.0!=vi) {
				++npop;
			}
		}
		return npop;
	}
	
	public int nzIndex() {
		for(int ii=0;ii<values.length;++ii) {
			final double vi = values[ii];
			if(0.0!=vi) {
				return indices[ii];
			}
		}
		return -1;
	}
	
	
	public double dot(final double[] x) {
		final int nindices = indices.length;
		double d = 0.0;
		for(int ii=0;ii<nindices;++ii) {
			d += values[ii]*x[indices[ii]];
		}
		return d;
	}
	
	
	/**
	 * slow (discouraged)
	 * @param i
	 * @return
	 */
	public double get(final int i) {
		if(i<0) {
			throw new ArrayIndexOutOfBoundsException(""+i);
		}
		final int ii;
		final int nindices = indices.length;
		if((i<nindices)&&(i==indices[i])) {
			// identity map, can skip binary search
			ii = i;
		} else {
			ii = Arrays.binarySearch(indices,i);
			if((ii<0)||(ii>=nindices)) {
				return 0.0;
			}
			if(i!=indices[ii]) {
				throw new IllegalStateException("lookup failed " + i + " -> " + ii + "\t" + indices[ii]);
			}
		}
		return values[ii];
	}
}
