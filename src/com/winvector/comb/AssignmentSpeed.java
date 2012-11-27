package com.winvector.comb;

import java.util.Random;

import org.apache.commons.math3.optimization.linear.SimplexSolver;

import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.ColtMatrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.apachem3.M3Solver;
import com.winvector.lp.apachem3.M3Solver.M3Prob;
import com.winvector.lp.glpk.GLPKSolver;
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
		final LPEQProb<NativeMatrix> prob = Assignment.buildAssignmentProb(NativeMatrix.factory,c);
		final LPEQProb<ColtMatrix> probSparse = Assignment.buildAssignmentProb(ColtMatrix.factory,c);
		// solve
		final int maxIts = 10000;
		final int nreps = 1;
		final LPSolver rSolver = new RevisedSimplexSolver();
		final M3Prob m3Prob = M3Solver.convertProbToM3(prob);
		final SimplexSolver m3solver = new SimplexSolver();
		final GLPKSolver glpkSolver = new GLPKSolver();
		m3solver.setMaxIterations(maxIts);
		LPSoln solnWV = null;
		LPSoln solnGLPK = null;
		double[] solnAPM3 = null;
		for(int rep=0;rep<nreps;++rep) {
			final double wvDTime;
			if(n<=75) {
				final long startWVD = System.currentTimeMillis();
				solnWV = rSolver.solve(prob,null,1.0e-5,maxIts);
				final long endWVD = System.currentTimeMillis();
				wvDTime = endWVD-startWVD;
			} else {
				wvDTime = Double.NaN;
			}
			final double wvSTime;
			if(n<=75) {
				final long startWVS = System.currentTimeMillis();
				rSolver.solve(probSparse,null,1.0e-5,maxIts);
				final long endWVS = System.currentTimeMillis();
				wvSTime = endWVS-startWVS;
			} else {
				wvSTime = Double.NaN;
			}
			final double apTime;
			if(n<=45) {
				final long startApache = System.currentTimeMillis();
				solnAPM3 = m3Prob.solve(m3solver);
				final long endApache = System.currentTimeMillis();
				apTime = endApache-startApache;
			} else {
				apTime = Double.NaN;
			}
			final double glpkTime;
			{
				final long startGLPK = System.currentTimeMillis();
				solnGLPK = glpkSolver.solve(prob,null,1.0e-5,maxIts);	
				final long endGLPK = System.currentTimeMillis();
				glpkTime = (endGLPK-startGLPK);
			}
			System.out.println("" + n + "\t" + wvDTime  + "\t" + wvSTime + "\t" + apTime + "\t" + glpkTime);
		}
		if(solnWV!=null) {
			final int[] lres = Assignment.computeAssignment(c,maxIts);
			if(!Assignment.checkValid(c,lres)) {
				throw new RuntimeException("bad solution");
			}
			final double costWV = Matrix.dot(solnWV.x,prob.c);
			double costGLPK = 0.0;
			for(int i=0;i<prob.c.length;++i) {
				costGLPK += solnGLPK.x[i]*prob.c[i];
			}
			if(Math.abs(costWV-costGLPK)>1.0e-3) {
				throw new Exception("solution costs did not match");
			}
			if(null!=solnAPM3) {
				double costAPM3 = 0.0;
				for(int i=0;i<prob.c.length;++i) {
					costAPM3 += solnAPM3[i]*prob.c[i];
				}
				if(Math.abs(costWV-costAPM3)>1.0e-3) {
					throw new Exception("solution costs did not match");
				}
			}
		}
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("" + "assignmentSize" + "\t" + "wvDenseTimeMS" + "\t" + "wvSparseTimeMS" + "\t" + "apm3TimeMS" + "\t" + "glpkTimeMS");
		for(int n=1;n<=100;++n) {
			for(int rep=0;rep<3;++rep) {
				runBoth(n);
			}
		}
	}
}
