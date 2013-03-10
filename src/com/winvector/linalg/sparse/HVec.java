package com.winvector.linalg.sparse;

import java.io.Serializable;
import java.util.Arrays;

import com.winvector.linalg.PreVecI;


/**
 * finite set of possible non-zero entries of a possibly infinite vector
 * @author johnmount
 *
 */
public class HVec implements Serializable {
	private static final long serialVersionUID = 1L;
	private static final double epsilon = 1.0e-12;
	
	final int[] indices;    // do not alter!
	final double[] values;  // do not alter!
	
	public HVec(final int[] indices, final double[] values) {
		final int origvlen = values.length;
		if(indices.length!=origvlen) {
			throw new IllegalArgumentException("indices.length=" + indices.length 
					+ ", values.length=" + values.length);
		}
		
		int nnz = 0;
		if(origvlen>0) {
			if(indices[0]<0) {
				throw new IllegalArgumentException("negative index");
			}
			for(int i=0;i<origvlen-1;++i) {
				if(indices[i+1]<=indices[i]) {
					throw new IllegalArgumentException("disordered indices");
				}
			}
		}
		for(final double vi: values) {
			if(Math.abs(vi)>epsilon) {
				++nnz;
			}
		}
		if(nnz<origvlen) {
			this.indices = new int[nnz];
			this.values = new double[nnz];
			int k = 0;
			for(int i=0;i<origvlen;++i) {
				final double vi = values[i];
				if(Math.abs(vi)>epsilon) {
					this.indices[k] = indices[i];
					this.values[k] = vi;
					++k;
					if(k>=nnz) {
						break;
					}
				}
			}
		} else {
			this.indices = indices;
			this.values = values;
		}
	}
	
	HVec(final double[] x) {
		final int dim = x.length;
		int index = 0;
		for(int i=0;i<dim;++i) {
			if(Math.abs(x[i])>epsilon) {
				++index;
			}
		}
		final int nnz = index;
		indices = new int[nnz];
		values = new double[nnz];
		index = 0;
		for(int i=0;(i<dim)&&(index<nnz);++i) {
			if(Math.abs(x[i])>epsilon) {
				indices[index] = i;
				values[index] = x[i];
				++index;
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
		return new HVec(x);
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
		return indices.length;
	}
	
	public int nzIndex() {
		if(indices.length<=0) {
			return -1;
		} else {
			return indices[0];
		}
	}
	
	
	public double dot(final double[] x) {
		final int nindices = indices.length;
		double d = 0.0;
		for(int ii=0;ii<nindices;++ii) {
			d += values[ii]*x[indices[ii]];
		}
		return d;
	}
	
	public double dot(final PreVecI x) {
		final int nindices = indices.length;
		double d = 0.0;
		for(int ii=0;ii<nindices;++ii) {
			d += values[ii]*x.get(indices[ii]);
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

	public void toArray(final double[] x) {
		Arrays.fill(x,0.0);
		final int nindices = indices.length;
		for(int ii=0;ii<nindices;++ii) {
			x[indices[ii]] = values[ii];
		}
	}
	
	public double[] toArray(final int columns) {
		final double[] x = new double[columns];
		final int nindices = indices.length;
		for(int ii=0;ii<nindices;++ii) {
			x[indices[ii]] = values[ii];
		}
		return x;
	}
}
