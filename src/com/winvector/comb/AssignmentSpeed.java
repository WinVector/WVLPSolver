package com.winvector.comb;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.apachem3.M3Solver;
import com.winvector.lp.glpk.GLPKSolver;
import com.winvector.lp.impl.RevisedSimplexSolver;

public class AssignmentSpeed {

	/**
	 * can remove solvers once they are slow
	 * @param prob
	 * @param solvers
	 * @return
	 * @throws LPException
	 */
	public static Map<String,Long> runSet(final LPEQProb prob, Map<String,LPSolver> solvers) throws LPException {
		final Map<String,Long> res = new TreeMap<String,Long>();
		final Set<String> zaps = new HashSet<String>();
		for(final Map.Entry<String,LPSolver> me: solvers.entrySet()) {
			final String name = me.getKey();
			final LPSolver solver = me.getValue();
			double sawValue = Double.NaN;
			if(null!=solver) {
				LPSoln soln = null;
				final long startMS = System.currentTimeMillis();
				try {
					soln = solver.solve(prob,null,1.0e-5,100000);
				} catch (LPException e) {
				}
				final long endMS = System.currentTimeMillis();
				final long durationMS = endMS-startMS;
				if(null!=soln) {
					LPEQProb.checkPrimFeas(prob.A, prob.b, soln.x, 1.0e-3);
					final double value = Matrix.dot(soln.x,prob.c);
					if(!Double.isNaN(sawValue)) {
						if(Math.abs(value-sawValue)>1.0e-3) {
							throw new LPErrorException("solution costs did not match");
						}
					} else {
						sawValue = value;
					}
					if(soln.basis!=null) {
						final double[] dual = prob.inspectForDual(soln,1.0e-3);
						LPEQProb.checkPrimDualOpt(prob.A, prob.b, prob.c, soln.x, dual, 1.0e-3);
					}
					res.put(name,durationMS);
				}
				if((null==soln)||(durationMS>=5000)) {
					zaps.add(name);
				}
			}
		}
		for(final String zap: zaps) {
			solvers.put(zap,null);
		}
		return res;
	}
	

	public static void main(String[] args) throws Exception {
		final Random rand = new Random(235135L);
		final Map<String,LPSolver> solvers = new TreeMap<String,LPSolver>();
		solvers.put("ApacheM3Simplex",new M3Solver());
		solvers.put("WVLPSolver",new RevisedSimplexSolver());
		solvers.put("GLPK",new GLPKSolver());
		System.out.print("assignmentSize");
		for(final String name: solvers.keySet()) {
			System.out.print("\t" + name);
		}
		System.out.println();
		for(int n=5;n<=80;n+=5) {
			for(int rep=0;rep<3;++rep) {
				final double[][] c = new double[n][n];
				for(int i=0;i<n;++i) {
					for(int j=0;j<n;++j) {
						c[i][j] = rand.nextDouble();
					}
				}
				final LPEQProb prob = Assignment.buildAssignmentProb(NativeMatrix.factory,c);
				final Map<String,Long> durations = runSet(prob,solvers);
				System.out.print(n);
				for(final String name: solvers.keySet()) {
					final Long val = durations.get(name);
					System.out.print("\t" + ((val!=null)?val:"NaN"));
				}
				System.out.println();
			}
		}
	}
}
