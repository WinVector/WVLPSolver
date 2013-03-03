package com.winvector.lp.apachem3;

import java.util.ArrayList;

import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.PointValuePair;
import org.apache.commons.math3.optimization.linear.LinearConstraint;
import org.apache.commons.math3.optimization.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optimization.linear.Relationship;
import org.apache.commons.math3.optimization.linear.SimplexSolver;

import com.winvector.linagl.HVec;
import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;

/**
 * use Apache math 3 to solve LP
 * @author johnmount
 *
 * @param <T>
 */
public final class M3Solver implements LPSolver {
	
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

	public static M3Prob convertProbToM3(final LPEQProb p) {
		M3Prob r = new M3Prob();
		final int m = p.A.rows();
		final int n = p.A.cols();
		for(int i=0;i<m;++i) {
			final double[] coef = new double[n];
			for(int j=0;j<n;++j) {
				coef[j] = p.A.get(i, j);
			}
			final LinearConstraint lc = new LinearConstraint(coef,Relationship.EQ,p.b[i]);
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
	public <T extends Matrix<T>> LPSoln solve(final LPEQProb prob, final int[] basis_in, final double tol,
			final int maxRounds, final LinalgFactory<T> factory) throws LPException {
		final M3Prob m3Prob = convertProbToM3(prob);
		final SimplexSolver m3solver = new SimplexSolver();
		m3solver.setMaxIterations(maxRounds);
		final double[] s = m3Prob.solve(m3solver);
		final double[] solnVec = new double[s.length];
		for(int i=0;i<s.length;++i) {
			solnVec[i] = s[i];
		}
		final LPSoln r = new LPSoln(HVec.hVec(solnVec),null,null);
		return r;
	}

}
