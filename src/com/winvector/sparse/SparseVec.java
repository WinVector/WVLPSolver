package com.winvector.sparse;

import java.io.Serializable;
import java.util.Arrays;

import com.winvector.linagl.Matrix;

/**
 * immutable vector, pop count can not be taking from length (some zero entries may be stored)
 * @author johnmount
 *
 */
public final class SparseVec implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final int dim;
	int[] indices;    // do not alter!
	double[] values;  // do not alter!
	
	public SparseVec(final int dim, final int coord, final double val) {
		if((coord<0)||(coord>dim)) {
			throw new IllegalArgumentException();
		}
		this.dim = dim;
		indices = new int[1];
		values = new double[1];
		indices[0] = coord;
		values[0] = val;
	}
	
	public SparseVec(final double[] x) {
		dim = x.length;
		int index = 0;
		for(int i=0;i<dim;++i) {
			if(x[i]!=0) {
				++index;
			}
		}
		final boolean forceDense = (dim<5)||(index>=dim)||(index>=(2*dim)/3);
		final int nnz = forceDense?dim:index;
		indices = new int[nnz];
		values = new double[nnz];
		index = 0;
		for(int i=0;(i<dim)&&(index<nnz);++i) {
			if(forceDense||(x[i]!=0)) {
				indices[index] = i;
				values[index] = x[i];
				++index;
			}
		}
	}
	
	SparseVec(final SparseVec o, final double[] scale) {
		dim = o.dim;
		indices = o.indices; // share indices
		values = new double[indices.length];
		for(int ii=0;ii<indices.length;++ii) {
			final int i = o.indices[ii];
			indices[ii] = i;
			if(null==scale) {
				values[ii] = o.values[ii];
			} else {
				values[ii] = o.values[ii]*scale[i];
			}
		}
	}
	
	public SparseVec(final int dim, final int[] indices, final double[] values) {
		this.dim = dim;
		this.indices = indices;
		this.values = values;
		if(indices.length>0) {
			if(indices[0]<0) {
				throw new IllegalArgumentException("negative index");
			}
			if(indices[indices.length-1]>=dim) {
				throw new IllegalArgumentException("out of bounds index");
			}
			for(int i=0;i<indices.length-1;++i) {
				if(indices[i+1]<=indices[i]) {
					throw new IllegalArgumentException("disordered indices");
				}
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
	
	public double[] toDense() {
		final double[] x = new double[dim];
		for(int ii=0;ii<values.length;++ii) {
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
		final int ii;
		if((i<indices.length)&&(i==indices[i])) {
			// identity map, can skip binary search
			ii = i;
		} else {
			ii = Arrays.binarySearch(indices,i);
			if((ii<0)||(ii>=indices.length)) {
				return 0.0;
			}
			if(i!=indices[ii]) {
				throw new IllegalStateException("lookup failed " + i + " -> " + ii + "\t" + indices[ii]);
			}
		}
		return values[ii];
	}
}
