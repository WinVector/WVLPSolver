package com.winvector.lp;

import java.io.PrintStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import com.winvector.linagl.ColumnMatrix;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.PreMatrix;
import com.winvector.lp.LPException.LPMalformedException;

abstract class LPProbBase implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final ColumnMatrix A;
	public final double[] b;
	public final double[] c;
	public final String relStr;
	

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c
	 *            n-vector
	 * @throws LPException.LPMalformedException
	 *             if parameters don't match defs
	 */
	public LPProbBase(final ColumnMatrix A_in, final double[] b_in, final double[] c_in, final String relStr)
			throws LPException.LPMalformedException {
		checkParams(A_in, b_in, c_in);
		A = A_in;
		b = b_in;
		c = c_in;
		this.relStr = relStr;
	}
	
	/**
	 * @param A_
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c_
	 *            n-vector
	 * @throws LPException.LPMalformedException
	 *             if parameters don't match defs
	 */
	public static void checkParams(final PreMatrix A_, final double[] b_, final double[] c_)
			throws LPException.LPMalformedException {
		if ((A_ == null) || (b_ == null) || (c_ == null)
				|| (A_.rows() != b_.length) || (A_.cols() != c_.length)) {
			String problem = "misformed problem";
			if (A_ == null) {
				problem = problem + " A_==null";
			}
			if (b_ == null) {
				problem = problem + " b==null";
			} else {
				if (A_ != null) {
					if (A_.rows() != b_.length) {
						problem = problem + " A_.rows()(" + A_.rows()
								+ ")!=b.length(" + b_.length + ")";
					}
				}
			}
			if (c_ == null) {
				problem = problem + " c_==null";
			} else {
				if ((A_ != null) && (A_.rows() > 0)) {
					if (A_.cols() != c_.length) {
						problem = problem + " A_.cols()(" + A_.cols()
								+ ")!=c.length(" + c_.length + ")";
					}
				}
			}
			throw new LPException.LPMalformedException(problem);
		}
	}

	public abstract LPEQProb eqForm() throws LPMalformedException;
	
	public void print(final PrintStream p) {
		p.println();
		p.println("x>=0");
		A.print(p);
		p.print(" * x " + relStr + " ");
		p.println(Matrix.toString(b));
		p.print("minimize x . ");
		p.println(Matrix.toString(c));
		p.println();
	}
	
	public void print() {
		print(System.out);
	}
	
	/**
	 * print out in CPLEX problem format
	 * @param p
	 */
	public void printCPLEX(final PrintStream p) {
		final NumberFormat vnf = new DecimalFormat("00000");
		final NumberFormat vvf = new DecimalFormat("#.######E0");
		p.println("\\* WVLPSovler com.winvector.lp.LPEQProb see: http://www.win-vector.com/blog/2012/11/yet-another-java-linear-programming-library/ *\\");
		p.println();
		p.println("Minimize");
		p.print("\tvalue: ");
		{
			boolean first = true;
			for(int j=0;j<c.length;++j) {
				final double cj = c[j];
				if(Math.abs(cj)!=0) {
					final String valCStr = vvf.format(cj);
					if(!first) {
						if(valCStr.charAt(0)!='-') {
							p.print(" +");
						} else {
							p.print(" ");
						}
					} else {
						first = false;
					}
					p.print(valCStr + " " + "x" + vnf.format(j));
				}
			}
		}
		p.println();
		p.println();
		p.println("Subject To");
		for(int i=0;i<b.length;++i) {
			int nnz = 0;
			for(int j=0;j<c.length;++j) {
				final double aij = A.get(i, j);
				if(Math.abs(aij)!=0) {
					++nnz;
				}
			}
			if(nnz>0) {
				p.print("\teq" + vnf.format(i) + ":\t");
				boolean first = true;
				for(int j=0;j<c.length;++j) {
					final double aij = A.get(i, j);
					if(Math.abs(aij)!=0) {
						final String valStr = vvf.format(aij);
						if(!first) {
							if(valStr.charAt(0)!='-') {
								p.print(" +");
							} else {
								p.print(" ");
							}
						} else {
							first = false;
						}
						p.print(valStr + " " + "x" + vnf.format(j));
					}
				}
				p.println(" " + relStr + " " + vvf.format(b[i]));
			}
		}
		p.println();
		p.println("Bounds");
		for(int j=0;j<c.length;++j) {
			p.println("\t0 <= x" + vnf.format(j));
		}
		p.println();
		p.println("End");
		p.println();
		p.println("\\* eof *\\");
	}

}
