package com.winvector.lp.lp_solve;

import java.io.File;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.PrintStream;
import java.lang.reflect.Field;

import com.winvector.linagl.Matrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.glpk.GLPKSolver;

public class LP_solve implements LPSolver {
	private final String lp_solve_dir = "/Users/johnmount/Documents/workspace/WVLPSolver/lp_solve";

	private static final class ResConsumer extends Thread {
		public final LineNumberReader lnr;
		public final double[] v;
		
		public ResConsumer(final Process r, final int n) {
			lnr = new LineNumberReader(new InputStreamReader(r.getInputStream()));
			v = new double[n];
		}
		
		@Override
		public void run() {
			try {
				String line = "";
				while((line=lnr.readLine())!=null) {
					if(line.indexOf("Actual values of the variables:")>=0) {
						break;
					}
				}
				if(line!=null) {
					while((line=lnr.readLine())!=null) {
						final String[] flds = line.split("\\s+");
						if(flds.length==2) {
							final String varName = flds[0];
							final double value = Double.parseDouble(flds[1]);
							final int varIndex = Integer.parseInt(varName.substring(1));
							v[varIndex] = value;
						} else {
							break;
						}
					}
				}	
			} catch (Exception ex) {
				throw new RuntimeException(ex);
			}
		}
	}
	
	@Override
	public <T extends Matrix<T>> LPSoln solve(LPEQProb<T> prob, int[] basis_in,
			double tol, int maxRounds) throws LPException {
		try {
			final File tempFI = File.createTempFile("glpkProb",".txt");
			final File tempFS = File.createTempFile("glpkSoln",".txt");
			tempFI.delete();
			tempFS.delete();
			final PrintStream p = new PrintStream(tempFI);
			GLPKSolver.printCPLEX(prob,p);
			p.close();
			// ./lp_solve -rxli xli_CPLEX /var/folders/62/q44nj1wj2vq2qz0d_gljt3y80000gn/T/glpkProb4031416958578596285.txt -wbas res.txt
			final String oldPath = System.getProperty("java.library.path");
			if(oldPath.indexOf(lp_solve_dir)<0) {
				System.setProperty( "java.library.path", oldPath + ":" + lp_solve_dir );
				Field fieldSysPath = ClassLoader.class.getDeclaredField( "sys_paths" );
				fieldSysPath.setAccessible( true );
				fieldSysPath.set( null, null );
			}
			
			final String[] cmd = { "./lp_solve", "-rxli", "xli_CPLEX", tempFI.getAbsolutePath(), "-wbas", tempFS.getAbsolutePath() };
			final Process r = Runtime.getRuntime().exec(cmd,new String[]{},new File(lp_solve_dir));
			final ResConsumer consumer = new ResConsumer(r,prob.c.length);
			consumer.start();
			final int status = r.waitFor();
			if(status!=0) {
				throw new LPErrorException("glpk status: " + status);
			}
			consumer.join();
			tempFI.delete();
			tempFS.delete();
			return new LPSoln(consumer.v,null);
		} catch (Exception e) {
			throw new LPErrorException("glpk caught: " + e);
		}
	}

}
