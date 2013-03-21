package com.winvector.comb;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.jblas.JBlasMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.apachem3.M3Solver;
import com.winvector.lp.glpk.GLPKSolver;
import com.winvector.lp.impl.RevisedSimplexSolver;

public final class AssignmentSpeed {
	
	public static final class RunStats {
		public final Map<String,Long> res = new TreeMap<String,Long>();
		public final Set<String> zaps = new HashSet<String>();
		public long pivots = 0;
		public long inspections = 0;
		public long inspecionTimeMS = 0;
		public long totalTimeMS = 0;
		public long prePivotTimeMS = 0;
		public long postPivotTimeMS = 0;
	}

	/**
	 * @param prob
	 * @param solvers
	 * @return
	 * @throws LPException
	 */
	public static <T extends Matrix<T>> RunStats runSet(final LPEQProb prob, Map<String,LPSolver> solvers, final LinalgFactory<T> factory) throws LPException {
		final RunStats res = new RunStats();
		for(final Map.Entry<String,LPSolver> me: solvers.entrySet()) {
			final String name = me.getKey();
			final LPSolver solver = me.getValue();
			double sawValue = Double.NaN;
			if(null!=solver) {
				LPSoln soln = null;
				final long startTimeMS = System.currentTimeMillis();
				try {
					soln = solver.solve(prob,null,1.0e-5,100000,factory);
				} catch (LPException e) {
				}
				final long endTimeMS = System.currentTimeMillis();
				final long durationMS;
				if(null!=soln) {
					durationMS = soln.reportedRunTimeMS;
				} else {
					durationMS = endTimeMS - startTimeMS;
				}
				if(solver instanceof RevisedSimplexSolver) {
					final RevisedSimplexSolver rs = (RevisedSimplexSolver)solver;
					res.pivots = rs.pivots;
					res.inspections = rs.inspections;
					res.inspecionTimeMS = rs.inspectionTimeMS;
					res.totalTimeMS = rs.totalTimeMS;
					res.prePivotTimeMS = rs.prePivotTimeMS;
					res.postPivotTimeMS = rs.postPivotTimeMS;
					rs.clearCounters();
					final double[] dualSoln = prob.dualSolution(soln.basisColumns, factory);
					prob.checkPrimDualOpt(soln.primalSolution, dualSoln, 1.0e-5);
				}
				if(null!=soln) {
					prob.checkPrimFeas(soln.primalSolution, 1.0e-3);
					final double value = soln.primalSolution.dot(prob.c);
					if(!Double.isNaN(sawValue)) {
						if(Math.abs(value-sawValue)>1.0e-3) {
							throw new LPErrorException("solution costs did not match");
						}
					} else {
						sawValue = value;
					}
					if(soln.basisColumns!=null) {
						final double[] dual = prob.dualSolution(soln.basisColumns,factory);
						prob.checkPrimDualOpt(soln.primalSolution, dual, 1.0e-3);
					}
					res.res.put(name,durationMS);
				}
				if((null==soln)||(durationMS>=5000)) {
					res.zaps.add(name);
				}
			}
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
		System.out.print("\t" + "dim");
		System.out.print("\t" + "rows");
		System.out.print("\t" + "inspections");
		System.out.print("\t" + "pivots");
		System.out.print("\t" + "totTimeMS");
		System.out.print("\t" + "inspectionTimeMS");
		System.out.print("\t" + "prePivotTimeMS");
		System.out.print("\t" + "postPivotTimeMS");
		for(final String name: solvers.keySet()) {
			System.out.print("\t" + name);
		}
		System.out.println();
		for(int n=5;n<=80;n+=5) {
			final Set<String> zaps = new HashSet<String>();
			for(int rep=0;rep<3;++rep) {
				final double[][] c = new double[n][n];
				for(int i=0;i<n;++i) {
					for(int j=0;j<n;++j) {
						c[i][j] = rand.nextDouble();
					}
				}
				final LPEQProb prob = Assignment.buildAssignmentProb(JBlasMatrix.factory,c);
				final RunStats durations = runSet(prob,solvers,JBlasMatrix.factory);
				zaps.addAll(durations.zaps);
				System.out.print(n);
				System.out.print("\t" + prob.nvars());
				System.out.print("\t" + prob.rows());
				System.out.print("\t" + durations.inspections);
				System.out.print("\t" + durations.pivots);
				System.out.print("\t" + durations.totalTimeMS);
				System.out.print("\t" + durations.inspecionTimeMS);
				System.out.print("\t" + durations.prePivotTimeMS);
				System.out.print("\t" + durations.postPivotTimeMS);
				for(final String name: solvers.keySet()) {
					final Long val = durations.res.get(name);
					System.out.print("\t" + ((val!=null)?val:"NaN"));
				}
				System.out.println();
			}
			for(final String zap: zaps) {
				solvers.put(zap,null);
			}
		}
	}
}
