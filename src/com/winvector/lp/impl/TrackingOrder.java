package com.winvector.lp.impl;

import java.util.Arrays;
import java.util.Random;
import java.util.Set;

import com.winvector.lp.InspectionOrder;

public final class TrackingOrder implements InspectionOrder {
	
	private static final class InspectionRecord implements Comparable<InspectionRecord> {
		public final int id;
		public int nTimesTaken = 0;
		public int nTimesLiked = 0;
		public int nTimesDisliked = 0;
		public double sortKey = 0.0;
		
		public InspectionRecord(final int id) {
			this.id = id;
		}

		@Override
		public int compareTo(final InspectionRecord o) {
			if(nTimesLiked!=o.nTimesLiked) {
				if(nTimesLiked>=o.nTimesLiked) {
					return -1;
				} else {
					return 1;
				}
			}
			if(sortKey!=o.sortKey) {
				if(sortKey<=o.sortKey) {
					return -1;
				} else {
					return 1;
				}
			}
			if(id!=o.id) {
				if(id<=o.id) {
					return -1;
				} else {
					return 1;
				}
			}
			return 0;
		}
		
		@Override
		public boolean equals(final Object o) {
			return compareTo((InspectionRecord)o)==0;
		}
		
		@Override
		public int hashCode() {
			return id;
		}
		
		@Override
		public String toString() {
			return "" + id;
		}
	}
	
	private final Random rand;
	private final InspectionRecord[] origRecs;
	private final InspectionRecord[] workingRecs;
	private int ptr = 0;

	
	public TrackingOrder(final int n, final Random rand) {
		this.rand = rand;
		origRecs = new InspectionRecord[n];
		workingRecs = new InspectionRecord[n];
		for(int i=0;i<n;++i) {
			final InspectionRecord r = new InspectionRecord(i);
			origRecs[i] = r;
			workingRecs[i] = r;
			r.sortKey = rand.nextDouble();
		}
		shuffle();
	}
	
	@Override
	public boolean hasNext() {
		return ptr<workingRecs.length;
	}
	
	@Override
	public int take(final Set<Integer> current, final double[] lambda) {
		final InspectionRecord r = workingRecs[ptr];
		++ptr;
		++r.nTimesTaken;
		return r.id;
	}

	@Override
	public void startPass() {
		ptr = 0;
	}
	
	@Override
	public void shuffle() {
		ptr = 0;
		final int n = workingRecs.length;
		for(int i=0;i<n;++i) {
			workingRecs[i].sortKey = rand.nextDouble(); 
		}
		Arrays.sort(workingRecs);
	}

	@Override
	public void liked(final int i) {
		origRecs[i].nTimesLiked += 1;
	}

	@Override
	public void disliked(final int i) {
		origRecs[i].nTimesDisliked += 1;
	}
}
