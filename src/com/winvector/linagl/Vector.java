package com.winvector.linagl;

import java.io.PrintStream;
import java.io.Serializable;

public abstract class Vector implements Serializable {
	private static final long serialVersionUID = 1L;
	

	abstract public int size();
	abstract public double get(int i);
	abstract public void set(int i, double xpi);

	
	abstract public Vector newVector(int length);
	abstract public Vector copy();
	
	
	public int nextIndex(int x) {
		++x;
		final int n = size();
		while(x<n) {
			if(get(x)!=0.0) {
				return x;
			}
			++x;
		}
		return -1;
	}

	public double dot(final Vector x) {
		double r = 0.0;
		for(int i=0;i<size();++i) {
			r += get(i)*x.get(i);
		}
		return r;
	}

	public void print(final PrintStream p) {
		p.print('[');
		for (int i = 0; i < size(); ++i) {
			if (i > 0) {
				p.print(' ');
			}
			p.print(""+get(i));
		}
		p.print(']');
	}

	public Vector extract(final int[] basis) {
		final Vector r = newVector(basis.length);
		for (int col = 0; col < basis.length; ++col) {
			final double e = get(basis[col]);
			if (e!=0) {
				r.set(col, e);
			}
		}
		return r;
	}


	public int nNonZero() {
		int n = 0;
		for(int i=0;i<size();++i) {
			if(Math.abs(get(i))>0) {
				++n;
			}
		}
		return n;
	}

	public boolean isZero() {
		return nNonZero()<=0;
	}

	public double distSq(final Vector o) {
		double distSq = 0.0;
		for(int i=0;i<size();++i) {
			final double diffi = get(i) - o.get(i);
			distSq += diffi*diffi;
		}
		return distSq;
	}
	
	public void subtract(final Vector x) {
		for(int i=0;i<size();++i) {
			set(i,get(i)-x.get(i));
		}
	}
}

