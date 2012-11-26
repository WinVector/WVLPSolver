package com.winvector.lp.apachem3;

import java.util.Random;

import org.apache.commons.math3.optimization.linear.SimplexSolver;

import com.winvector.comb.Assignment;
import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.NativeLinAlg;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.apachem3.M3Solver.M3Prob;
import com.winvector.lp.impl.RevisedSimplexSolver;

public class AssignmentSpeed {

	private static void runBoth(final int n) throws Exception {
		final Random rand = new Random(235135L);
		final double[][] c = new double[n][n];
		for(int i=0;i<n;++i) {
			for(int j=0;j<n;++j) {
				c[i][j] = rand.nextDouble();
			}
		}
		final LPEQProb<NativeMatrix> prob = Assignment.buildAssignmentProb(NativeLinAlg.factory,c);
		// solve
		final int maxIts = 10000;
		final int nreps = 5;
		final LPSolver rSolver = new RevisedSimplexSolver();
		LPSoln<NativeMatrix> soln1 = null;
		// apache library
		final M3Prob m3Prob = M3Solver.convertProbToM3(prob);
		final SimplexSolver m3solver = new SimplexSolver();
		m3solver.setMaxIterations(maxIts);
		double[] soln2 = null;
		for(int rep=0;rep<nreps;++rep) {
			final long startWV = System.currentTimeMillis();
			soln1 = rSolver.solve(prob,null,1.0e-5,maxIts);
			final long endWV = System.currentTimeMillis();
			final long startApache = System.currentTimeMillis();
			soln2 = m3Prob.solve(m3solver);
			final long endApache = System.currentTimeMillis();
			final long wvTime = (endWV-startWV);
			final long apTime = (endApache-startApache);
			System.out.println("" + n + "\t" + wvTime + "\t" + apTime);
		}
		final int[] lres = Assignment.computeAssignment(c,maxIts);
		if(!Assignment.checkValid(c,lres)) {
			//System.out.println(new NativeMatrix(c));
			throw new RuntimeException("bad solution");
		}
		final double cost1 = Matrix.dot(soln1.x,prob.c);
		double cost2 = 0.0;
		for(int i=0;i<prob.c.length;++i) {
			cost2 += soln2[i]*prob.c[i];
		}
		if(Math.abs(cost1-cost2)>1.0e-3) {
			throw new Exception("solution costs did not match");
		}
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("" + "n" + "\t" + "wvTimeMS" + "\t" + "apm3TimeMS");
		for(int n=1;n<=40;++n) {
			for(int rep=0;rep<5;++rep) {
				runBoth(n);
			}
		}
	}
}
