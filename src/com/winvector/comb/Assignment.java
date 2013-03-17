package com.winvector.comb;

import java.util.BitSet;
import java.util.Random;

import com.winvector.linalg.DenseVec;
import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.colt.ColtMatrix;
import com.winvector.linalg.jblas.JBlasMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.lp.impl.RevisedSimplexSolver;

/**
 * assignment problem solved by a linear program.  mostly test/demo the lp code as you would want to solve actual assignment
 * problems in production with direct combinatorial code like: http://algs4.cs.princeton.edu/65reductions/AssignmentProblem.java.html
 * Find permutation of 0...n-1 such that sum_i cost(i,p(i)) is minimized and for all i cost(i,p(i)) is finite and not-NaN (so 
 * user can use NaN to signal non edges).   This is a minimual weight complete matching 
 * @author johnmount
 *
 */
public final class Assignment {
	private Assignment() {
	}
	
	/**
	 * 
	 * @param cost n b n matrix
	 * @return minimal cost total assignment, or null if there is no complete matching
	 * @throws LPMalformedException 
	 */
	public static <T extends Matrix<T>> LPEQProb buildAssignmentProb(final LinalgFactory<T> factory, final double[][] cost) throws LPMalformedException {
		final int n = cost.length;
		final double maxAbsVal;
		{
			final BitSet lhs = new BitSet(n);
			final BitSet rhs = new BitSet(n);
			double maxAbsSeen = 0.0;
			for(int i=0;i<n;++i) {
				for(int j=0;j<n;++j) {
					final double cij = cost[i][j];
					if((!Double.isNaN(cij))&&(!Double.isInfinite(cij))) {
						maxAbsSeen = Math.max(maxAbsSeen,Math.abs(cij));
						lhs.set(i);
						rhs.set(j);
					}
				}
			}
			maxAbsVal = maxAbsSeen;
			if((lhs.cardinality()!=n)||(rhs.cardinality()!=n)) {
				return null;
			}
		}
		final int m = 2*n;
		final int nVars = n*n;
		final double bigValue = (m+1)*(nVars+1)*(maxAbsVal+1.0);
		final T a = factory.newMatrix(m, nVars, true);
		final double[] b = new double[m];
		final double[] c = new double[nVars];
		{
			int nextIndex = 0;
			for(int i=0;i<n;++i) {
				for(int j=0;j<n;++j) {
					final double cij = (!Double.isNaN(cost[i][j]))&&(!Double.isInfinite(cost[i][j]))?cost[i][j]:bigValue;
					a.set(i,nextIndex,1.0);             // sum_i x_{i,j} = 1 for all j
					a.set(n+j,nextIndex,1.0);           // sum_i x_{i,j} = 1 for all j
					c[nextIndex] = cij;
					++nextIndex;
				}
			}
		}
		for(int i=0;i<m;++i) {
			b[i] = 1.0;
		}
		final LPEQProb prob = new LPEQProb(a.columnMatrix(),b,new DenseVec(c));
		return prob;
	}
	
	public static <T extends Matrix<T>> int[] computeAssignment(final double[][] cost, final LinalgFactory<T> factory, final LPSolver solver, final int maxIts) {
		final int n = cost.length;
		if(n<=0) {
			return new int[0];
		}
		if(n!=cost[0].length) {
			return null;
		}
		try {
			final LPEQProb prob = buildAssignmentProb(factory,cost);
			final LPSoln soln1 = solver.solve(prob, null, 1.0e-6,maxIts, factory);
			final int[] assignment = new int[n];
			int nextIndex = 0;
			for(int i=0;i<n;++i) {
				for(int j=0;j<n;++j) {
					final double cij = cost[i][j];
					if((!Double.isNaN(cij))&&(!Double.isInfinite(cij))) {
						final double sv = soln1.primalSolution.get(nextIndex); // TODO: avoid get()
						if(sv>=0.5) {
							assignment[i] = j;
						}						
					}
					++nextIndex;
				}
			}
			return assignment;
		} catch (LPException e) {
			//e.printStackTrace();
			return null;
		}
	}
	
	public static <T extends Matrix<T>> int[] computeAssignment(final double[][] cost, final int maxIts, final LinalgFactory<T> factory) {
		final RevisedSimplexSolver solver = new RevisedSimplexSolver();
		return computeAssignment(cost,factory,solver,maxIts);
	}
	
	public static boolean checkValid(final double[][] cost, final int[] assignment) {
		if(null==assignment) {
			return false;
		}
		final int n = cost.length;
		if(n!=assignment.length) {
			return false;
		}
		final BitSet saw = new BitSet(n);
		for(int i=0;i<n;++i) {
			if((assignment[i]<0)||(assignment[i]>=n)) {
				return false;
			}
			final double cij = cost[i][assignment[i]];
			if((Double.isNaN(cij))||(Double.isInfinite(cij))) {
				return false;
			}
			if(saw.get(i)) {
				return false;
			}
			saw.set(i);
		}
		if(saw.cardinality()!=n) {
			return false;
		}
		return true;
	}
	
	public static double cost(final double[][] cost, final int[] assignment) {
		double tot = 0.0;
		for(int i=0;i<cost.length;++i) {
			tot += cost[i][assignment[i]];
		}
		return tot;
	}
	
	public static void main(final String[] args) throws LPException {
		final Random rand = new Random(235135L);
		final int n = 200;
		final RevisedSimplexSolver solver = new RevisedSimplexSolver();
		final double[][] c = new double[n][n];
		for(int i=0;i<n;++i) {
			for(int j=0;j<n;++j) {
				c[i][j] = rand.nextDouble();
			}
		}
		final LPEQProb prob = Assignment.buildAssignmentProb(JBlasMatrix.factory,c);
		System.out.print("assignmentSize");
		System.out.print("\t" + "dim");
		System.out.print("\t" + "rows");
		System.out.print("\t" + "inspections");
		System.out.print("\t" + "pivots");
		System.out.print("\t" + "totTimeMS");
		System.out.print("\t" + "inspectionTimeMS");
		System.out.print("\t" + "prePivotTimeMS");
		System.out.print("\t" + "postPivotTimeMS");
		System.out.print("\t" + "solnTimeMS");
		System.out.println();
		while(true) {
			final long solnStart = System.currentTimeMillis();
			final LPSoln soln = solver.solve(prob,null,1.0e-5,100000,JBlasMatrix.factory);
			final long solnDone = System.currentTimeMillis();
			final long pivots = solver.pivots;
			final long inspections = solver.inspections;
			final long inspecionTimeMS = solver.inspectionTimeMS;
			final long totalTimeMS = solver.totalTimeMS;
			final long prePivotTimeMS = solver.prePivotTimeMS;
			final long postPivotTimeMS = solver.postPivotTimeMS;
			final long solnTimeMS = solnDone - solnStart;
			solver.clearCounters();
			System.out.print(n);
			System.out.print("\t" + prob.nvars());
			System.out.print("\t" + prob.rows());
			System.out.print("\t" + inspections);
			System.out.print("\t" + pivots);
			System.out.print("\t" + totalTimeMS);
			System.out.print("\t" + inspecionTimeMS);
			System.out.print("\t" + prePivotTimeMS);
			System.out.print("\t" + postPivotTimeMS);
			System.out.print("\t" + solnTimeMS);
			System.out.println();
		}
	}
}
