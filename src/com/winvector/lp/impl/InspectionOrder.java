package com.winvector.lp.impl;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

public final class InspectionOrder {
	
	private static final class InspectionRow implements Comparable<InspectionRow> {
		public final int index;
		public long randMark;
		public int nTakes = 0;
		
		public InspectionRow(final int index, final long randMark) {
			this.index = index;
			this.randMark = randMark;
		}

		@Override
		public int compareTo(final InspectionRow o) {
			// compare first on nTakes
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
	private final InspectionRow[] idToRow;
	private final ArrayList<InspectionRow> allRows;
	private final SortedSet<InspectionRow> available = new TreeSet<InspectionRow>();

	
	public InspectionOrder(final int n, final Random rand, final BitSet skips) {
		this.rand = rand;
		allRows = new ArrayList<InspectionRow>(n);
		idToRow = new InspectionRow[n];
		for(int i=0;i<n;++i) {
			if((null==skips)||(!skips.get(i))) {
				final InspectionRow r = new InspectionRow(i,rand.nextLong());
				allRows.add(r);
				idToRow[i] = r;
			}
		}
	}
	
	public boolean hasNext() {
		return !available.isEmpty();
	}
	
	public int take() {
		final InspectionRow r = available.first();
		available.remove(r);
		r.nTakes += 1;
		return r.index;
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
		for(final InspectionRow r: allRows) {
			r.randMark = rand.nextLong();
		}
		available.addAll(allRows);
	}
}
