package com.winvector.linalg;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;

import com.winvector.linalg.sparse.ColumnMatrix;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;




public abstract class Matrix<T extends Matrix<T>> implements Serializable {
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
	
	abstract public Object buildExtractTemps();
	abstract public SparseVec extractColumn(final int ci, final Object extractTemps);

	
	public ColumnMatrix columnMatrix() {		
		final int cols = cols();
		final SparseVec[] columns = new SparseVec[cols];
		final Object extractTemps = buildExtractTemps();
		for(int j=0;j<cols;++j) {
			columns[j] = extractColumn(j,extractTemps);
		}
		return new ColumnMatrix(rows(),columns);
	}
	
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



	public static double dot(final double[] x, final double[] y) {
		double r = 0.0;
		final int n = x.length;
		for(int i=0;i<n;++i) {
			r += x[i]*y[i];
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
	

	int firstNZCol(final int row, final double minVal, final BitSet usedColumns) {
		final int ccols = cols();
		for(int j=0;j<ccols;++j) {
			if(!usedColumns.get(j)) {
				final double cijAbs = Math.abs(get(row,j));
				if((cijAbs>0)&&(cijAbs>=minVal)) {
					return j;
				}
			}
		}
		return -1;
	}
	
	void clearCol(final int aggressorRow, final int victimRow, final int col) {
		final double avc = get(victimRow,col);
		if(avc!=0.0) {
			final double aac = get(aggressorRow,col);
			final double scale = avc/aac;
			final int ccols = cols();
			for(int j=0;j<ccols;++j) {
				set(victimRow,j,get(victimRow,j)-scale*get(aggressorRow,j));
			}
			set(victimRow,col,0.0); // get rid of some rounding error
		}
	}
	
	void rescaleRow(final int row, final int col) {
		final double arc = get(row,col);
		final double scale = 1.0/arc;
		final int ccols = cols();
		for(int j=0;j<ccols;++j) {
			set(row,j,get(row,j)*scale);
		}
		set(row,col,1.0); // get rid of some rounding error
	}
	
	private void elimBasis(final int[] foundRow, final int[] foundCol, final int row) {
		for(int ii=0;ii<foundRow.length;++ii) {
			final int r = foundRow[ii];
			if(r<0) {
				break;
			}
			final int c =  foundCol[ii];
			clearCol(r,row,c);
		}
	}
	
	private void addBasis(final int nFound, final int[] foundRow, final int[] foundCol,
			final BitSet usedColumns,
			final int newRow, final int newCol) {
		rescaleRow(newRow,newCol);
		foundRow[nFound] = newRow;
		foundCol[nFound] = newCol;
		usedColumns.set(newCol);
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
		final int crows = c.rows();
		final int ccols = c.cols();
		final int nGoal = Math.min(ccols,crows);
		final BitSet rowsSeen = new BitSet(crows);
		final BitSet usedColumns = new BitSet(ccols);
		final int[] foundRow = new int[nGoal];
		final int[] foundCol = new int[nGoal];
		Arrays.fill(foundRow,-1);
		Arrays.fill(foundCol,-1);
		int nFound = 0;
		if(null!=forcedRows) {
			for(final int ri: forcedRows) {
				rowsSeen.set(ri);
				c.elimBasis(foundRow,foundCol,ri);
				final int j = c.firstNZCol(ri,minVal,usedColumns);
				if(j<0) {
					throw new IllegalArgumentException("candidate rows were not independent");	
				}
				c.addBasis(nFound,foundRow,foundCol,usedColumns,ri,j);
				++nFound;
			}
		}
		if(nFound<nGoal) {
			for(int ri=0;ri<crows;++ri) {
				if(!rowsSeen.get(ri)) {
					c.elimBasis(foundRow,foundCol,ri);
					final int j = c.firstNZCol(ri,minVal,usedColumns);
					if(j>=0) {
						c.addBasis(nFound,foundRow,foundCol,usedColumns,ri,j);
						++nFound;
						if(nFound>=nGoal) {
							break;
						}
					}
				}
			}
		}
		final int[] r = new int[nFound];
		for(int i=0;i<nFound;++i) {
			r[i] = foundRow[i];
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
}
