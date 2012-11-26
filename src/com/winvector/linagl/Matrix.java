package com.winvector.linagl;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;



public abstract class Matrix<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;
	

	abstract public int cols();
	abstract public int rows();
	abstract public boolean sparseRep();
	abstract public LinalgFactory<T> factory();

	abstract public double get(int j, int i);
	abstract public void set(int i, int j, double d);

	abstract public T copy();
	abstract public T transpose();

	abstract public <Z extends T> T multMat(final Z o);
	abstract public double[] solve(final double[] y, final boolean leastsq);
	abstract public T inverse();



	public double[] extractRow(final int ri) {
		final double[] r = new double[cols()];
		for(int i=0;i<cols();++i) {
			final double e = get(ri, i);
			r[i] = e;
		}
		return r;
	}
	
	public static double[] extract(final double[] v, final int[] indices) {
		final double[] r = new double[indices.length];
		for(int i=0;i<indices.length;++i) {
			r[i] = v[indices[i]];
		}
		return r;
	}

	public static boolean isZero(final double[] x) {
		for(final double xi: x) {
			if(Math.abs(xi)>0.0) {
				return false;
			}
		}
		return true;
	}

	public static int nNonZero(final double[] x) {
		int n = 0;
		for(final double xi: x) {
			if(Math.abs(xi)>0.0) {
				++n;
			}
		}
		return n;
	}

	public static double dot(final double[] x, final double[] y) {
		double r = 0.0;
		final int n = x.length;
		for(int i=0;i<n;++i) {
			r += x[i]*y[i];
		}
		return r;
	}
	
	public static double distSq(final double[] x, final double[] y) {
		double r = 0.0;
		final int n = x.length;
		for(int i=0;i<n;++i) {
			final double diffi = x[i]-y[i];
			r += diffi*diffi;
		}
		return r;
	}
	
	public static String toString(final double[] x) {
		final StringBuilder b = new StringBuilder();
		b.append("{");
		boolean first = true;
		for(final double xi: x) {
			if(!first) {
				b.append(", ");
			} else {
				first = false;
			}
			b.append(""+xi);
		}
		b.append("}");
		return b.toString();
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append("[" + rows() + "][" + cols() + "]{\n");
		for(int i=0;i<rows();++i) {
			b.append(" ");
			b.append(toString(extractRow(i)));
			if(i<rows()-1) {
				b.append(",");
			}
			b.append("\n");
		}
		b.append("}\n");
		return b.toString();
	}
	
	public void print(final PrintStream p) {
		p.print("{");
		p.println();
		for (int i = 0; i < rows(); ++i) {
			p.print(" ");
			p.println(toString(extractRow(i)));
			p.println();
		}
		p.print("}");
		p.println();
	}

	public <Z extends Matrix<Z>> Z extractColumns(final int[] basis, final LinalgFactory<Z> zfactory) {
		final Z r = zfactory.newMatrix(rows(), basis.length, false);
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

	public double[] mult(final double[] x) {
		if(cols()!=x.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[rows()];
		for (int i = 0; i < r.length; ++i) {
			double z = 0;
			for(int k=0;k<cols();++k) {
				z += x[k]*get(i,k);
			}
			r[i] = z;
		}
		return r;		
	}

	public double[] multLeft(final double[] b) {
		if (rows() != b.length) {
			throw new IllegalArgumentException();
		}
		final double[] r = new double[cols()];
		for (int i = 0; i < r.length; ++i) {
			double z = 0;
			for(int k=0;k<rows();++k) {
				z += b[k]*get(k, i);
			}
			r[i] = z;
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
	private static <Z extends Matrix<Z>> int largestNormRow(final Z c, final BitSet candidates, final double minNormSq) {
		int bestProbe = -1;
		double largestNormSq = Double.NEGATIVE_INFINITY;
		for(int probe=candidates.nextSetBit(0); probe>=0; probe=candidates.nextSetBit(probe+1)) {
			final double normSq = rowDotRow(c,probe,probe); 
			if(normSq>minNormSq) {
				if((bestProbe<0)||(normSq>largestNormSq)) {
					bestProbe = probe;
					largestNormSq = normSq;
				}
			} else {
				candidates.clear(probe);
			}
		}
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


	/**
	 * destructie to c
	 * @param c
	 * @param forcedRows
	 * @param minVal
	 * @return
	 */
	private static <Z extends Matrix<Z>> int[] rowBasis(final Z c, final int[] forcedRows, final double minVal) {
		final double minNormSq = minVal*minVal;
		final BitSet found = new BitSet(c.rows());
		if(null!=forcedRows) {
			final BitSet candidates = new BitSet(c.rows());
			for(final int ri: forcedRows) {
				candidates.set(ri);
			}
			while(!candidates.isEmpty()) {
				final int want = largestNormRow(c,candidates,minNormSq);
				if(want<0) {
					break;
				}
				found.set(want);
				elimRow(c,want);
				candidates.clear(want);
			}
			if(found.cardinality()!=forcedRows.length) {
				throw new IllegalArgumentException("candidate rows were not independent");
			}
		}
		// extend basis to the rest of the rows
		final BitSet candidates = new BitSet(c.rows());
		for(int i=0;i<c.rows();++i) {
			if(!found.get(i)) {
				candidates.set(i);
			}
		}
		while(!candidates.isEmpty()) {
			final int want = largestNormRow(c,candidates,minNormSq);
			if(want<0) {
				break; // exhausted possibilities
			}
			found.set(want);
			elimRow(c,want);
			candidates.clear(want);
		}
		final int[] r = new int[found.cardinality()];
		int i = 0;
		for(int fi=found.nextSetBit(0); fi>=0; fi=found.nextSetBit(fi+1)) {
			r[i] = fi;
			++i;
		}
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

	public void setRow(final int ri, final double[] col) {
		for(int i=0;i<cols();++i) {
			final double e = col[i];
			set(ri,i, e);
		}
	}

	public double[] extractColumn(final int ci) {
		final double[] r = new double[rows()];
		for(int i=0;i<rows();++i) {
			final double e = get(i, ci);
			r[i] = e;
		}
		return r;
	}

	public <Z extends Matrix<Z>> Z extractRows(final int[] rowset, final LinalgFactory<Z> zfactory) {
		final Z r = zfactory.newMatrix(rowset.length,cols(),sparseRep());
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
}
