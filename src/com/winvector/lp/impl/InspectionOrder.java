package com.winvector.lp.impl;

import java.util.ArrayList;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

public final class InspectionOrder {
	
	private static final class InspectionRow implements Comparable<InspectionRow> {
		public final int index;
		public long randMark;
		public int nTakes = 0;
		public int nLasts = 0;
		
		public InspectionRow(final int index, final long randMark) {
			this.index = index;
			this.randMark = randMark;
		}

		@Override
		public int compareTo(final InspectionRow o) {
			// compare first on nLasts
			if(nLasts!=o.nLasts) {
				if(nLasts<=o.nLasts) {
					return -1;
				} else {
					return 1;
				}
			}
			if(nTakes!=o.nTakes) {
				if(nTakes<=o.nTakes) {
					return -1;
				} else {
					return 1;
				}
			}
			if(randMark!=o.randMark) {
				if(randMark<=o.randMark) {
					return -1;
				} else {
					return 1;
				}
			}
			if(index!=o.index) {
				if(index<=o.index) {
					return -1;
				} else {
					return 1;
				}
			}
			return 0;
		}
		
		@Override
		public boolean equals(final Object o) {
			return index==((InspectionRow)o).index;
		}
		
		@Override
		public int hashCode() {
			return (int)randMark;
		}
	}
	
	private final Random rand;
	private final ArrayList<InspectionRow> rows;
	private final SortedSet<InspectionRow> available = new TreeSet<InspectionRow>();
	
	public static int[] perm(final Random rand, final int n) {
		final int[] perm = new int[n];
		for(int i=0;i<n;++i) {
			perm[i] = i;
		}
		for(int i=0;i<n-1;++i) {
			final int j = i + rand.nextInt(n-i);
			if(j>i) {
				final int vi = perm[i];
				final int vj = perm[j];
				perm[i] = vj;
				perm[j] = vi;
			}
		}
		return perm;
	}

	
	public InspectionOrder(final int n, final Random rand) {
		this.rand = rand;
		rows = new ArrayList<InspectionRow>(n);
		for(int i=0;i<n;++i) {
			final InspectionRow r = new InspectionRow(i,rand.nextLong());
			rows.add(r);
		}
	}
	
	public int take() {
		final InspectionRow r = available.first();
		available.remove(r);
		r.nTakes += 1;
		r.randMark = rand.nextLong();
		return r.index;
	}

	public void moveToEnd(final int v) {
		final InspectionRow r = rows.get(v);
		available.remove(r);
		r.nLasts += 1;
		r.randMark = rand.nextLong();
	}
	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		boolean first = true;
		for(final InspectionRow r: available) {
			if(!first) {
				b.append(" ");
			} else {
				first = false;
			}
			b.append(r.index);
		}
		return b.toString();
	}


	public void startPass() {
		available.clear();
		available.addAll(rows);
	}
}
