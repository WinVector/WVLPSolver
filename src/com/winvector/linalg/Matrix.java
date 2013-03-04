package com.winvector.linalg;

import java.io.PrintStream;
import java.util.BitSet;




public abstract class Matrix<T extends Matrix<T>> implements PreMatrixI {
	private static final long serialVersionUID = 1L;
	

	abstract public int cols();
	abstract public int rows();
	abstract public boolean sparseRep();
	abstract public LinalgFactory<T> factory();

	abstract public double get(int i, int j);
	abstract public void set(int i, int j, double d);

	abstract public <Z extends T> T multMat(final Z o);
	abstract public double[] solve(final double[] y);
	abstract public T inverse();
	
	
	public <Z extends Matrix<Z>> Z copy(final LinalgFactory<Z> factory, final boolean wantSparse) {
		final int rows = rows();
		final int cols = cols();
		final Z r = factory.newMatrix(rows,cols,wantSparse);
		for(int i=0;i<rows;++i) {
			for(int j=0;j<cols;++j) {
				final double vij = get(i,j);
				if(vij!=0) {
					r.set(i, j, vij);
				}
			}
		}
		return r;
	}
	
	public <Z extends Matrix<Z>> Z matrixCopy(final LinalgFactory<Z> factory) {
		return copy(factory,sparseRep());
	}

	public <Z extends Matrix<Z>> Z transpose(final LinalgFactory<Z> factory, final boolean wantSparse) {
		final int rows = rows();
		final int cols = cols();
		final Z r = factory.newMatrix(cols,rows,wantSparse);
		for(int i=0;i<rows;++i) {
			for(int j=0;j<cols;++j) {
				final double vij = get(i,j);
				if(vij!=0) {
					r.set(j, i, vij);
				}
			}
		}
		return r;
	}

	private double[] extractRow(final int ri) {
		final int cols = cols();
		final double[] r = new double[cols];
		for(int i=0;i<cols;++i) {
			final double e = get(ri, i);
			r[i] = e;
		}
		return r;
	}
	
	public static double[] extract(final double[] v, final int[] indices) {
		final int ilength = indices.length;
		final double[] r = new double[ilength];
		for(int i=0;i<ilength;++i) {
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
			if(xi!=0.0) {
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
		final int blength = basis.length;
		final int rows = rows();
		final Z r = zfactory.newMatrix(rows, blength, false);
		for (int col = 0; col < blength; ++col) {
			for (int row = 0; row < rows; ++row) {
				final double e = get(row, basis[col]);
				if (e!=0) {
					r.set(row, col, e);
				}
			}
		}
		return r;
	}

	@Override
	public double[] mult(final double[] x) {
		final int cols = cols();
		if(cols!=x.length) {
			throw new IllegalArgumentException();
		}
		final int rows = rows();
		final double[] r = new double[rows];
		for(int k=0;k<cols;++k) {
			final double xk = x[k];
			if(Math.abs(xk)>1.0e-8) {
				for (int i = 0; i < rows; ++i) {
					r[i] += xk*get(i,k);
				}
			}
		}
		return r;		
	}
	
	@Override
	public double[] mult(final HVec x) {
		final int rows = rows();
		final double[] r = new double[rows];
		final int nindices = x.nIndices();
		for(int ii=0;ii<nindices;++ii) {
			final int k = x.index(ii);
			final double xk = x.value(ii);
			if(Math.abs(xk)>1.0e-8) {
				for (int i = 0; i < rows; ++i) {
					r[i] += xk*get(i,k);
				}
			}
		}
		return r;		
	}

	@Override
	public double[] multLeft(final double[] b) {
		final int rows = rows();
		if (rows != b.length) {
			throw new IllegalArgumentException();
		}
		final int cols = cols();
		final double[] r = new double[cols];
		for(int k=0;k<rows;++k) {
			final double bk = b[k];
			if(Math.abs(bk)>1.0e-8) {
				for (int i = 0; i < cols; ++i) {
					r[i] += bk*get(k, i);
				}
			}
		}
		return r;
	}
	

	private static <Z extends Matrix<Z>> double rowDotRow(final Z c, final int a, final int b) {
		double bdotb = 0.0;
		final int dim = c.cols();
		for(int i=0;i<dim;++i) {
			bdotb += c.get(a,i)*c.get(b,i);
		}
		return bdotb;
	}
	

	
	/**
	 * elim all of elims from target
	 * @param c
	 * @param elims (target not in this set)
	 * @param target
	 */
	private static <Z extends Matrix<Z>> void elimRows(final Z c, final BitSet elims, final int target) {
		final double tdott = rowDotRow(c,target,target);
		if(tdott>0) {
			final int ccols = c.cols();
			for(int ei=elims.nextSetBit(0); ei>=0; ei=elims.nextSetBit(ei+1)) {
				if(ei!=target) {
					final double edote = rowDotRow(c,ei,ei);
					if(edote>0) {
						final double edott = rowDotRow(c,ei,target);
						if(edott!=0.0) {
							final double scale = edott/edote;
							for(int i=0;i<ccols;++i) {
								c.set(target,i,c.get(target,i) - c.get(ei,i)*scale);
							}
						}
					}
				}			
			}
		}
	}


	/**
	 * picks rows in order given (skipping rows in span of others)
	 * destructive to c
	 * @param c
	 * @param forcedRows
	 * @param minVal
	 * @return
	 */
	private static <Z extends Matrix<Z>> int[] rowBasis(final Z c, final int[] forcedRows, final double minVal) {
		final double minNormSq = minVal*minVal;
		final int crows = c.rows();
		final BitSet checked = new BitSet(crows);
		final BitSet found = new BitSet(crows);
		final int ccols = c.cols();
		final int nGoal = Math.min(ccols,crows);
		int nFound = 0;
		if(null!=forcedRows) {
			for(final int ri: forcedRows) {
				checked.set(ri);
				elimRows(c,found,ri);
				final double bdotb = rowDotRow(c,ri,ri);
				if((bdotb>0)&&(bdotb>=minNormSq)) {
					found.set(ri);
					++nFound;
					if(nFound>=nGoal) {
						break;
					}
				}
			}
			if(nFound!=forcedRows.length) {
				throw new IllegalArgumentException("candidate rows were not independent");
			}
		}
		for(int ri=0;(ri<crows)&&(nFound<nGoal);++ri) {
			if(!checked.get(ri)) {
				elimRows(c,found,ri);
				final double bdotb = rowDotRow(c,ri,ri);
				if((bdotb>0)&&(bdotb>=minNormSq)) {
					found.set(ri);
					++nFound;
				}
			}
		}
		final int[] r = new int[nFound];
		int i = 0;
		for(int fi=found.nextSetBit(0); fi>=0; fi=found.nextSetBit(fi+1)) {
			r[i] = fi;
			++i;
		}
		return r;
	}
	
	/**
	 * picks rows in order given (skipping rows in span of others)
	 * @param minVal
	 * @return
	 */
	public int[] rowBasis(final double minVal) {
		final T c = copy(factory(),false);
		return rowBasis(c,null,minVal);
	}
	
	/**
	 * picks columns in order given (skipping columns in span of others)
	 * @param forcedRows
	 * @param minVal
	 * @return
	 */
	public int[] colBasis(final int[] forcedRows, final double minVal) {
		final T c = transpose(factory(),false);
		return rowBasis(c,forcedRows,minVal);
	}

	public void setRow(final int ri, final double[] row) {
		final int cols = cols();
		for(int i=0;i<cols;++i) {
			final double e = row[i];
			set(ri,i, e);
		}
	}
	

	@Override
	public Object buildExtractTemps() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * 
	 * @param ci
	 * @return
	 */
	public SparseVec extractColumn(final int ci, final Object extractTemps) {
		final int rows = rows();
		final double[] r = new double[rows];
		for(int i=0;i<rows;++i) {
			final double e = get(i, ci);
			r[i] = e;
		}
		return SparseVec.sparseVec(r);
	}

	public <Z extends Matrix<Z>> Z extractRows(final int[] rowset, final LinalgFactory<Z> zfactory) {
		final int rowsetlength = rowset.length;
		final int cols = cols();
		final Z r = zfactory.newMatrix(rowsetlength,cols,sparseRep());
		for (int row = 0; row < rowsetlength; ++row) {
			for (int col = 0; col < cols; ++col) {
				final double e = get(rowset[row],col);
				if (e!=0) {
					r.set(row, col, e);
				}
			}
		}
		return r;
	}
}
