package com.winvector.comb;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Vector;
import com.winvector.linalg.colt.NativeLinAlg;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.impl.RevisedSimplexSolver;

/**
 * assignment problem
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
	public static LPEQProb<NativeMatrix> buildAssignmentProb(final double[][] cost) throws LPMalformedException {
		final int n = cost.length;
		final LinalgFactory<NativeMatrix> factory = NativeLinAlg.factory;
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
		final NativeMatrix a = factory.newMatrix(m, nVars, false);
		final Vector b = factory.newVector(m);
		final Vector c = factory.newVector(nVars);
		{
			int nextIndex = 0;
			for(int i=0;i<n;++i) {
				for(int j=0;j<n;++j) {
					final double cij = (!Double.isNaN(cost[i][j]))&&(!Double.isInfinite(cost[i][j]))?cost[i][j]:bigValue;
					a.set(i,nextIndex,1.0);             // sum_i x_{i,j} = 1 for all j
					a.set(n+j,nextIndex,1.0);           // sum_i x_{i,j} = 1 for all j
					c.set(nextIndex,cij);
					++nextIndex;
				}
			}
		}
		for(int i=0;i<m;++i) {
			b.set(i,1.0);
		}
		final LPEQProb<NativeMatrix> prob = new LPEQProb<NativeMatrix>(a,b,c);
		return prob;
	}
	
	public static int[] computeAssignment(final double[][] cost, final int maxIts) {
		final int n = cost.length;
		if(n<=0) {
			return new int[0];
		}
		if(n!=cost[0].length) {
			return null;
		}
		try {
			final LPEQProb<NativeMatrix> prob = buildAssignmentProb(cost);
			final LPSoln<NativeMatrix> soln1 = new RevisedSimplexSolver<NativeMatrix>().solve(prob, null, 1.0e-6,maxIts);
			final int[] assignment = new int[n];
			int nextIndex = 0;
			for(int i=0;i<n;++i) {
				for(int j=0;j<n;++j) {
					final double cij = cost[i][j];
					if((!Double.isNaN(cij))&&(!Double.isInfinite(cij))) {
						final double sv = soln1.x.get(nextIndex);
						if(sv>=0.5) {
							assignment[i] = j;
						}						
					}
					++nextIndex;
				}
			}
			return assignment;
		} catch (LPException e) {
			return null;
		}
	}
	
	public static boolean checkValid(final double[][] cost, final int[] assignment) {
		final int n = cost.length;
		if(n!=assignment.length) {
			return false;
		}
		final Set<Integer> saw = new HashSet<Integer>();
		for(int i=0;i<n;++i) {
			if((assignment[i]<0)||(assignment[i]>=n)) {
				return false;
			}
			final double cij = cost[i][assignment[i]];
			if((Double.isNaN(cij))||(Double.isInfinite(cij))) {
				return false;
			}
			if(saw.contains(i)) {
				return false;
			}
			saw.add(i);
		}
		if(saw.size()!=n) {
			return false;
		}
		return true;
	}
}
