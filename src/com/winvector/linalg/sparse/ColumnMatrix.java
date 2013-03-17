package com.winvector.linalg.sparse;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;

import com.winvector.linalg.PreMatrixI;

/**
 * immutable sparse matrix
 * @author johnmount
 *
 */
public final class ColumnMatrix implements PreMatrixI {
	private static final long serialVersionUID = 1L;

	private final int rows;
	private final int cols;
	private final SparseVec[] columns;
	
	
	public ColumnMatrix(final int rows, final SparseVec[] columns) {
		this.rows = rows;
		this.cols = columns.length;
		this.columns = columns;
	}
	
	
	private ColumnMatrix(final int rows, final int cols) {
		this.rows = rows;
		this.cols = cols;
		columns = new SparseVec[cols];
	}
	
	/**
	 * slow (discouraged)
	 * @param row
	 * @param col
	 * @return
	 */
	@Override
	public double get(final int row, final int col) {
		return columns[col].get(row);
	}
	
	@Override
	public SparseVec extractColumn(final int j) {
		return columns[j];
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append("[" + rows + "][" + cols + "]{\n");
		for(int i=0;i<rows;++i) {
			b.append(" {");
			for(int j=0;j<cols;++j) {
				if(j>0) {
					b.append(",");
				}
				b.append(" " + columns[j].get(i));
			}
			b.append("}");
			if(i<rows-1) {
				b.append(",");
			}
			b.append("\n");
		}
		b.append("}\n");
		return b.toString();
	}

	public void print(final PrintStream p) {
		p.println(toString());
	}


	@Override
	public double[] mult(final double[] x) {
		final double[] res = new double[rows];
		for(int j=0;j<cols;++j) {
			if(x[j]!=0) {
				final SparseVec col = columns[j];
				final int colindiceslength = col.indices.length;
				for(int ii=0;ii<colindiceslength;++ii) {
					final int i = col.indices[ii];
					final double aij = col.values[ii];
					res[i] += aij*x[j];
				}
			}
		}
		return res;
	}


	@Override
	public double[] multLeft(final double[] y) {
		final double[] res = new double[cols];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final int i = col.indices[ii];
				final double aij = col.values[ii];
				res[j] += aij*y[i];
			}
		}
		return res;
	}
	
	
	@Override
	public ColumnMatrix extractColumns(final int[] basis) {
		final int newCols = basis.length;
		final ColumnMatrix newMatrix = new ColumnMatrix(rows,newCols);
		for(int j=0;j<newCols;++j) {
			newMatrix.columns[j] = columns[basis[j]];
		}
		return newMatrix;
	}

	@Override
	public ColumnMatrix addColumns(final ArrayList<SparseVec> cs) {
		final int cssize = cs.size();
		final ColumnMatrix r = new ColumnMatrix(rows,cols+cssize);
		for(int i=0;i<cols;++i) {
			r.columns[i] = columns[i];
		}
		for(int i=0;i<cssize;++i) {
			if(cs.get(i).dim!=rows) {
				throw new IllegalArgumentException();
			}
			r.columns[cols+i] = cs.get(i);
		}
		return r;
	}
	
	@Override
	public double[] sumAbsRowValues() {
		final double[] r = new double[rows];
		for(int j=0;j<cols;++j) {
			final SparseVec col = columns[j];
			final int colindiceslength = col.indices.length;
			for(int ii=0;ii<colindiceslength;++ii) {
				final double aij = col.values[ii];
				if(0.0!=aij) {
					final int i = col.indices[ii];
					r[i] += Math.abs(aij);
				}
			}
		}
		return r;
	}
	
	@Override
	public ColumnMatrix extractRows(final int[] rb) {
		final int newDim = rb.length;
		final BitSet rows = new BitSet(rows());
		for(final int ri: rb) {
			rows.set(ri);
		}
		final ColumnMatrix newMat = new ColumnMatrix(newDim,cols);
		for(int j=0;j<cols;++j) {
			newMat.columns[j] = columns[j].extractRows(newDim, rows);
		}
		return newMat;
	}

	@Override
	public ColumnMatrix rescaleRows(double[] scale) {
		final ColumnMatrix r = new ColumnMatrix(rows,cols);
		for(int j=0;j<cols;++j) {
			r.columns[j] = columns[j].scale(scale);
		}
		return r;
	}

	@Override
	public int rows() {
		return rows;
	}

	@Override
	public int cols() {
		return cols;
	}

	@Override
	public double[] mult(final HVec x) {
		final int rows = rows();
		final double[] r = new double[rows];
		final int nColIndices = x.indices.length;
		for(int jj=0;jj<nColIndices;++jj) {
			final int k = x.indices[jj];
			final double xk = x.values[jj];
			if(Math.abs(xk)>1.0e-8) {
				final SparseVec col = columns[k];
				final int nRowIndices = col.nIndices();
				for(int ii=0;ii<nRowIndices;++ii) {
					final int i = col.index(ii);
					final double vij = col.value(ii);
					r[i] += xk*vij;
				}
			}
		}
		return r;		
	}
	
	
	
	
	private static int firstNZRow(final double[] col, final double minVal, final BitSet usedRows) {
		final int nrows = col.length;
		for(int i=0;i<nrows;++i) {
			if(!usedRows.get(i)) {
				final double coliAbs = Math.abs(col[i]);
				if((coliAbs>0)&&(coliAbs>=minVal)) {
					return i;
				}
			}
		}
		return -1;
	}
	
	private static void elimBasis(final SparseVec[] basisCols, final int[] foundRow, final double[] col) {
		final int nf = basisCols.length;
		for(int jj=0;jj<nf;++jj) {
			final SparseVec v = basisCols[jj];
			if(null==v) {
				break;
			}
			final int r = foundRow[jj];
			final double scale = -col[r];
			if(Math.abs(scale)>0) {
				final int nr = v.nIndices();
				for(int ii=0;ii<nr;++ii) {
					final int i = v.index(ii);
					final double vi = v.value(ii);
					col[i] += scale*vi;
				}
			}
			col[r] = 0.0;  //smash some floating point error
		}
	}
	
	private static SparseVec addBasis(final int nFound, final double[] workCol,
			final int[] foundRow, final int[] foundCol,
			final BitSet usedRows,
			final int newRow, final int newCol) { 
		final SparseVec nC = SparseVec.sparseVec(workCol).scale(1.0/workCol[newRow]);
		foundRow[nFound] = newRow;
		foundCol[nFound] = newCol;
		usedRows.set(newRow);
		return nC;
	}
	
	/**
	 * picks rows in order given (skipping rows in span of others)
	 * @param forcedCols
	 * @param minVal
	 * @return
	 */
	public int[] colBasis(final int[] forcedCols, final double minVal) {
		final int nGoal = Math.min(cols,rows);
		final BitSet colsSeen = new BitSet(cols);
		final BitSet usedRows = new BitSet(rows);
		final int[] foundRow = new int[nGoal];
		final int[] foundCol = new int[nGoal];
		final SparseVec[] basisCols = new SparseVec[nGoal];
		final double[] workCol = new double[rows];
		Arrays.fill(foundRow,-1);
		Arrays.fill(foundCol,-1);
		int nFound = 0;
		if(null!=forcedCols) {
			for(final int cj: forcedCols) {
				colsSeen.set(cj);
				columns[cj].toArray(workCol);
				elimBasis(basisCols,foundRow,workCol);
				final int i = firstNZRow(workCol,minVal,usedRows);
				if(i<0) {
					throw new IllegalArgumentException("candidate cols were not independent");	
				}
				basisCols[nFound] = addBasis(nFound,workCol,foundRow,foundCol,usedRows,i,cj);
				++nFound;
			}
		}
		if(nFound<nGoal) {
			for(int cj=0;cj<cols;++cj) {
				if(!colsSeen.get(cj)) {
					columns[cj].toArray(workCol);
					elimBasis(basisCols,foundRow,workCol);
					final int i = firstNZRow(workCol,minVal,usedRows);
					if(i>=0) {
						basisCols[nFound] = addBasis(nFound,workCol,foundRow,foundCol,usedRows,i,cj);
						++nFound;
						if(nFound>=nGoal) {
							break;
						}
					}
				}
			}
		}
		final int[] b = new int[nFound];
		for(int j=0;j<nFound;++j) {
			b[j] = foundCol[j];
		}
		return b;
	}
	
	@Override
	public ColumnMatrix transpose() {
		// TODO: implement
		// count number of indices in each row
		final int[] nc = new int[rows];
		for(int cj=0;cj<cols;++cj) {
			final SparseVec col = columns[cj];
			final int nr = col.nIndices();
			for(int ii=0;ii<nr;++ii) {
				final int i = col.index(ii);
				nc[i] += 1;
			}
		}
		// alloc them and copy values in
		final int[][] nindices = new int[rows][];
		final double[][] nvalues = new double[rows][];
		for(int i=0;i<rows;++i) {
			nindices[i] = new int[nc[i]];
			nvalues[i] = new double[nc[i]];
		}
		Arrays.fill(nc,0);
		for(int cj=0;cj<cols;++cj) {
			final SparseVec col = columns[cj];
			final int nr = col.nIndices();
			for(int ii=0;ii<nr;++ii) {
				final int i = col.index(ii);
				final double v = col.value(ii);
				nindices[i][nc[i]] = cj;
				nvalues[i][nc[i]] = v;
				nc[i] += 1;
			}
		}
		// copy out to matrix structure
		final ColumnMatrix t = new ColumnMatrix(cols,rows);
		for(int i=0;i<rows;++i) {
			t.columns[i] = new SparseVec(cols,nindices[i],nvalues[i]);
		}
		return t;
	}
}
