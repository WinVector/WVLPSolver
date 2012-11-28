package com.winvector.lp.glpk;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import com.winvector.linagl.Matrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;

/**
 * call GLPK solver through file system (involves expensive parsing/un-parsing of results)
 * @author johnmount
 *
 */
public class GLPKSolver implements LPSolver {
	private final String glpksolverPath = "/opt/local/bin/glpsol";
	
	/**
	 * print out in CPLEX problem format
	 * @param p
	 */
	public static <T extends Matrix<T>> void printCPLEX(final LPEQProb<T> prob, final PrintStream p) {
		final NumberFormat vnf = new DecimalFormat("00000");
		final NumberFormat vvf = new DecimalFormat("#.######E0");
		p.println("\\* WVLPSovler com.winvector.lp.LPEQProb see: http://www.win-vector.com/blog/2012/11/yet-another-java-linear-programming-library/ *\\");
		p.println();
		p.println("Minimize");
		p.print("\tvalue: ");
		{
			boolean first = true;
			for(int j=0;j<prob.c.length;++j) {
				final double cj = prob.c[j];
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
		for(int i=0;i<prob.b.length;++i) {
			int nnz = 0;
			for(int j=0;j<prob.c.length;++j) {
				final double aij = prob.A.get(i, j);
				if(Math.abs(aij)!=0) {
					++nnz;
				}
			}
			if(nnz>0) {
				p.print("\teq" + vnf.format(i) + ":\t");
				boolean first = true;
				for(int j=0;j<prob.c.length;++j) {
					final double aij = prob.A.get(i, j);
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
				p.println(" = " + vvf.format(prob.b[i]));
			}
		}
		p.println();
		p.println("Bounds");
		for(int j=0;j<prob.c.length;++j) {
			p.println("\t0 <= x" + vnf.format(j));
		}
		p.println();
		p.println("End");
		p.println();
		p.println("\\* eof *\\");
	}

	@Override
	public <T extends Matrix<T>> LPSoln solve(final LPEQProb<T> prob, final int[] basis_in,
			final double tol, final int maxRounds) throws LPException {
		try {
			final File tempFI = File.createTempFile("glpkProb",".txt");
			final File tempFS = File.createTempFile("glpkSoln",".txt");
			tempFI.delete();
			tempFS.delete();
			final PrintStream p = new PrintStream(tempFI);
			printCPLEX(prob,p);
			p.close();
			// glpsol --lp tmp.txt -w soln.txt
			final String[] cmd = { glpksolverPath, "--lp", tempFI.getAbsolutePath(), "-w", tempFS.getAbsolutePath() };
			final Process r = Runtime.getRuntime().exec(cmd);
			final int status = r.waitFor();
			if(status!=0) {
				throw new LPErrorException("glpk status: " + status);
			}
			final LineNumberReader lnr = new LineNumberReader(new FileReader(tempFS));
			final String header = lnr.readLine();
			final int m = Integer.parseInt(header.split(" ")[0]);
			lnr.readLine(); // throw out descriptive line
			for(int i=0;i<m;++i) { // throw out constraint lines
				lnr.readLine();
			}
			final double[] v = new double[prob.c.length];
			for(int i=0;i<v.length;++i) {
				final String vline = lnr.readLine();
				v[i] = Double.parseDouble(vline.split("\\s+")[1]);
			}
			tempFI.delete();
			tempFS.delete();
			return new LPSoln(v,null);
		} catch (IOException e) {
			throw new LPErrorException("glpk caught: " + e);
		} catch (InterruptedException e) {
			throw new LPErrorException("glpk caught: " + e);
		}
	}

}
