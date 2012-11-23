package com.winvector.lp.apachem3;

import java.util.ArrayList;

import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.PointValuePair;
import org.apache.commons.math3.optimization.linear.LinearConstraint;
import org.apache.commons.math3.optimization.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optimization.linear.Relationship;
import org.apache.commons.math3.optimization.linear.SimplexSolver;

import com.winvector.linagl.Matrix;
import com.winvector.linagl.Vector;
import com.winvector.lp.LPException;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;

/**
 * use Apache math 3 to solve LP
 * @author johnmount
 *
 * @param <T>
 */
public final class M3Solver<T extends Matrix<T>> implements LPSolver<T> {
	
	public static final class M3Prob {
		public final ArrayList<LinearConstraint> constraints = new ArrayList<LinearConstraint>();
		public GoalType goalType = GoalType.MINIMIZE;
		public boolean restrictToNonNegative = true;
		public LinearObjectiveFunction f = null;
		
		public double[] solve(final SimplexSolver m3solver) {
			final PointValuePair soln = m3solver.optimize(f,constraints,goalType,restrictToNonNegative);
			final double[] s = soln.getPoint();
			return s;
		}
	}

	public static <Z extends Matrix<Z>> M3Prob convertProbToM3(final LPEQProb<Z> p) {
		M3Prob r = new M3Prob();
		final int m = p.A.rows();
		final int n = p.A.cols();
		for(int i=0;i<m;++i) {
			final double[] coef = new double[n];
			for(int j=0;j<n;++j) {
				coef[j] = p.A.get(i, j);
			}
			final LinearConstraint lc = new LinearConstraint(coef,Relationship.EQ,p.b.get(i));
			r.constraints.add(lc);
		}
		final double[] obj = new double[n];
		for(int j=0;j<n;++j) {
			obj[j] = p.c.get(j);
		}
		r.f =  new LinearObjectiveFunction(obj,0.0);
		return r;
	}
	

	@Override
	public LPSoln<T> solve(final LPEQProb<T> prob, final int[] basis_in, final double tol,
			final int maxRounds) throws LPException {
		final M3Prob m3Prob = convertProbToM3(prob);
		final SimplexSolver m3solver = new SimplexSolver();
		final double[] s = m3Prob.solve(m3solver);
		final Vector solnVec = prob.A.newVector(s.length);
		for(int i=0;i<s.length;++i) {
			solnVec.set(i,s[i]);
		}
		final LPSoln<T> r = new LPSoln<T>(solnVec,null);
		return r;
	}

}
