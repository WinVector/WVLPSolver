package com.winvector.comb;

import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.LPSolver;
import com.winvector.lp.apachem3.M3Solver;
import com.winvector.lp.glpk.GLPKSolver;
import com.winvector.lp.impl.RevisedSimplexSolver;
import com.winvector.lp.lp_solve.LP_solve;

public class GeomSpeed {


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
	

	
	public static void main(String[] args) throws Exception {
		final Random rand = new Random(235135L);
		final Map<String,LPSolver> solvers = new TreeMap<String,LPSolver>();
		solvers.put("lp_solve",new LP_solve());
		solvers.put("ApacheM3Simplex",new M3Solver());
		solvers.put("WVLPSolver",new RevisedSimplexSolver());
		solvers.put("GLPK",new GLPKSolver());
		System.out.print("geomDimension");
		for(final String name: solvers.keySet()) {
			System.out.print("\t" + name);
		}
		System.out.println();
		for(int n=5;n<=80;n+=5) {
			for(int rep=0;rep<3;++rep) {
				final LPEQProb<NativeMatrix> prob = randGeomProb(n,6*n+4, rand); 
				final Map<String,Long> durations = AssignmentSpeed.runSet(prob,solvers);
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
