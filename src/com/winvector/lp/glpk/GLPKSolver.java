package com.winvector.lp.glpk;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintStream;

import com.winvector.linagl.LinalgFactory;
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
	
	@Override
	public <T extends Matrix<T>> LPSoln solve(final LPEQProb prob, final int[] basis_in,
			final double tol, final int maxRounds, final  LinalgFactory<T> factory) throws LPException {
		try {
			final File tempFI = File.createTempFile("glpkProb",".txt");
			final File tempFS = File.createTempFile("glpkSoln",".txt");
			tempFI.delete();
			tempFS.delete();
			final PrintStream p = new PrintStream(tempFI);
			prob.printCPLEX(p);
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
