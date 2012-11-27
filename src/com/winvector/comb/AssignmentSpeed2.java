package com.winvector.comb;

import java.util.Random;

import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.glpk.GLPKSolver;
import com.winvector.lp.impl.RevisedSimplexSolver;

public class AssignmentSpeed2 {


	/**
	 * 
	 * @param dim
	 * @param facets
	 * @param rand
	 * @return bounded feasible problem
	 * @throws LPMalformedException 
	 */
	public static LPEQProb<NativeMatrix> randGeomProb(final int dim, final int facets, final Random rand) throws LPMalformedException {
		final int n = dim + facets;   // one slack var per facet
		final int m = facets;
		final Matrix<NativeMatrix> a = NativeMatrix.factory.newMatrix(m,n,false);
		final double[] b = new double[m];
		final double[] c = new double[n];
		// objective only on original variables
		for(int i=0;i<dim;++i) {
			c[i] = rand.nextGaussian();
		}
		// first dim relens are x_i <= 1, which is x_i+ s_i = 1, so we are bounded in a cube
		for(int i=0;i<dim;++i) {
			a.set(i,i,1.0);
			a.set(i,dim+i,1.0);
			b[i] = 1.0;
		}
		// rest of the relations are random cuts with the point (0.5,...,0.5) strictly in the interior
		for(int i=dim;i<facets;++i) {
			while(true) {
				// want a.x <= 1 or a.x >=1 (depending on what a.(0.5,...,0.5) does
				double dot = 0;
				for(int j=0;j<dim;++j) {
					final double aij = rand.nextGaussian();
					a.set(i,j,aij);
					dot += 0.5*aij;
				}
				if(Math.abs(dot-1.0)>1/(double)(n+m+1.0)) {
					if(dot<=1.0) {
						// a.x <= 1 => a.x + s = 1
						a.set(i,dim+i,1.0);
						b[i] = 1.0;
					} else {
						// a.x >= 1 => a.x - s = 1
						a.set(i,dim+i,-1.0);
						b[i] = 1.0;
					}
					break;
				}
			}
		}
		return new LPEQProb<NativeMatrix>(a,b,c);
	}
	
	private static void runBoth(final int n) throws LPException {
		final Random rand = new Random(235135L);
		final LPEQProb<NativeMatrix> prob = randGeomProb(n,6*n+4,rand);
		// solve
		final int maxIts = 100000;
		final int nreps = 1;
		final RevisedSimplexSolver rSolver = new RevisedSimplexSolver();
		final GLPKSolver glpkSolver = new GLPKSolver();
		for(int rep=0;rep<nreps;++rep) {
			final double wvDTime;
			final LPSoln solnWV;
			final double[] dualWV;
			try {
				final long startWVD = System.currentTimeMillis();
				solnWV = rSolver.solve(prob,null,1.0e-5,maxIts);
				final long endWVD = System.currentTimeMillis();
				wvDTime = endWVD-startWVD;
				dualWV = prob.inspectForDual(solnWV, 1.0e-5);
				LPEQProb.checkPrimDualOpt(prob.A, prob.b, prob.c, solnWV.x, dualWV, 1.0e-4); // throws if not both feasible and optimal
			} catch (LPException ex) {
				System.out.println("problem with wv");
				GLPKSolver.printCPLEX(prob, System.out);
				throw ex;
			}
			final double glpkTime;
			final LPSoln solnGLPK;
			try {
				final long startGLPK = System.currentTimeMillis();
				solnGLPK = glpkSolver.solve(prob,null,1.0e-5,maxIts);	
				final long endGLPK = System.currentTimeMillis();
				glpkTime = (endGLPK-startGLPK);
			} catch (LPException ex) {
				System.out.println("problem with glpk");
				GLPKSolver.printCPLEX(prob, System.out);
				System.out.println("WV soln:");
				System.out.println("primal: " + solnWV);
				System.out.println("dual: " + Matrix.toString(dualWV));
				throw ex;
			}
			System.out.println("" + n + "\t" + wvDTime + "\t" + glpkTime);
		}
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("" + "problemSize" + "\t" + "wvDenseTimeMS" + "\t" + "glpkTimeMS");
		for(int n=10;n<=100;n+=10) {
			for(int rep=0;rep<3;++rep) {
				runBoth(n);
			}
		}
	}
}
