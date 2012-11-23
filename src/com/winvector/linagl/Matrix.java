package com.winvector.linagl;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;



public abstract class Matrix<T extends Matrix<T>> implements LinalgFactory<T> {
	abstract public int cols();
	abstract public int rows();
	abstract public boolean sparseRep();

	abstract public double get(int j, int i);
	abstract public void set(int i, int j, double d);

	abstract public T copy();
	abstract public T transpose();

	abstract public <Z extends T> T multMat(final Z o);
	abstract public Vector solve(final Vector y, final boolean leastsq);
	abstract public T inverse();



	public Vector extractRow(final int ri) {
		final Vector r = newVector(cols());
		for(int i=0;i<cols();++i) {
			final double e = get(ri, i);
			if (e!=0) {
				r.set(i, e);
			}
		}
		return r;
	}

	public void print(final PrintStream p) {
		p.print('[');
		p.println();
		for (int i = 0; i < rows(); ++i) {
			final Vector ri = extractRow(i);
			p.print(' ');
			ri.print(p);
			p.println();
		}
		p.print(']');
		p.println();
	}

	public T extractColumns(final int[] basis) {
		final T r = newMatrix(rows(), basis.length, false);
		for (int col = 0; col < basis.length; ++col) {
			for (int row = 0; row < rows(); ++row) {
				final double e = get(row, basis[col]);
				if (e!=0) {
					r.set(row, col, e);
				}
			}
		}
		return r;
	}

	public Vector mult(final Vector x) {
		if(cols()!=x.size()) {
			throw new IllegalArgumentException();
		}
		final Vector r = newVector(rows());
		for (int i = 0; i < r.size(); ++i) {
			double z = 0;
			for(int k=0;k<cols();++k) {
				z += x.get(k)*get(i,k);
			}
			if (z!=0) {
				r.set(i, z);
			}
		}
		return r;		
	}

	public Vector multLeft(final Vector b) {
		if (rows() != b.size()) {
			throw new IllegalArgumentException();
		}
		final Vector r = newVector(cols());
		for (int i = 0; i < r.size(); ++i) {
			double z = 0;
			for(int k=0;k<rows();++k) {
				z += b.get(k)*get(k, i);
			}
			if (z!=0) {
				r.set(i, z);
			}
		}
		return r;
	}
	

	private static <Z extends Matrix<Z>> double rowDotRow(final Z c, final int a, final int b) {
		double bdotb = 0.0;
		for(int i=0;i<c.cols();++i) {
			bdotb += c.get(a,i)*c.get(b,i);
		}
		return bdotb;
	}
	
	/**
	 * 
	 * @param c
	 * @param candidates can alter candidates (remove zero rows)
	 * @return
	 */
	private static <Z extends Matrix<Z>> int largestNormRow(final Z c, final Set<Integer> candidates, final double minNormSq) {
		int bestProbe = -1;
		double largestNormSq = Double.NEGATIVE_INFINITY;
		final Set<Integer> victims = new HashSet<Integer>(2*candidates.size()+5);
		for(final Integer probe: candidates) {
			final double normSq = rowDotRow(c,probe,probe); 
			if(normSq>minNormSq) {
				if((bestProbe<0)||(normSq>largestNormSq)) {
					bestProbe = probe;
					largestNormSq = normSq;
				}
			} else {
				victims.add(probe);
			}
		}
		candidates.removeAll(victims);
		return bestProbe;
	}
	
	private static <Z extends Matrix<Z>> void elimRow(final Z c, final int row) {
		final double bdotb = rowDotRow(c,row,row);
		if(bdotb>0) {
			for(int elim=0;elim<c.rows();++elim) {
				if(row!=elim) {
					final double bdotx = rowDotRow(c,row,elim);
					if(Math.abs(bdotx)>0) {
						for(int i=0;i<c.cols();++i) {
							c.set(elim,i,c.get(elim,i) - c.get(row,i)*bdotx/bdotb);
						}
					}
				}
			}
			for(int i=0;i<c.cols();++i) {
				c.set(row,i,0.0);
			}
		}
	}


	public static <Z extends Matrix<Z>> int[] rowBasis(final Z c, final int[] forcedRows, final double minVal) {
		final double minNormSq = minVal*minVal;
		final Set<Integer> found = new HashSet<Integer>();
		if(null!=forcedRows) {
			final Set<Integer> candidates = new HashSet<Integer>(2*forcedRows.length+5);
			for(final int ri: forcedRows) {
				candidates.add(ri);
			}
			while(!candidates.isEmpty()) {
				final int want = largestNormRow(c,candidates,minNormSq);
				if(want<0) {
					break;
				}
				found.add(want);
				elimRow(c,want);
				candidates.remove(want);
			}
			if(found.size()!=forcedRows.length) {
				throw new IllegalArgumentException("candidate rows were not independent");
			}
		}
		// extend basis to the rest of the rows
		final Set<Integer> candidates = new HashSet<Integer>(2*c.rows()+5);
		for(int i=0;i<c.rows();++i) {
			if(!found.contains(i)) {
				candidates.add(i);
			}
		}
		while(!candidates.isEmpty()) {
			final int want = largestNormRow(c,candidates,minNormSq);
			if(want<0) {
				break; // exhausted possibilities
			}
			found.add(want);
			elimRow(c,want);
			candidates.remove(want);
		}
		final int[] r = new int[found.size()];
		int i = 0;
		for(final int fi: found) {
			r[i] = fi;
			++i;
		}
		Arrays.sort(r);
		return r;
	}
	
	public int[] rowBasis(final int[] forcedRows, final double minVal) {
		final T c = copy();
		return rowBasis(c,forcedRows,minVal);
	}
	
	public int[] colBasis(final int[] forcedRows, final double minVal) {
		final T c = transpose();
		return rowBasis(c,forcedRows,minVal);
	}

	public void setRow(final int ri, final Vector col) {
		for(int i=0;i<cols();++i) {
			final double e = col.get(i);
			set(ri,i, e);
		}
	}

	public Vector extractColumn(final int ci) {
		final Vector r = newVector(rows());
		for(int i=0;i<rows();++i) {
			final double e = get(i, ci);
			if (e!=0) {
				r.set(i, e);
			}
		}
		return r;
	}

	public T extractRows(final int[] rowset) {
		final T r = newMatrix(rowset.length,cols(),sparseRep());
		for (int row = 0; row < rowset.length; ++row) {
			for (int col = 0; col < cols(); ++col) {
				final double e = get(rowset[row],col);
				if (e!=0) {
					r.set(row, col, e);
				}
			}
		}
		return r;
	}
		
	public T identityMatrix(final int m, boolean wantSparse) {
		final T r = newMatrix(m,m,wantSparse);
		for(int i=0;i<m;++i) {
			r.set(i, i, 1.0);
		}
		return r;
	}
}
