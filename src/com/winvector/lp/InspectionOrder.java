package com.winvector.lp;

import java.util.Set;

public interface InspectionOrder {
	void startPass();
	void shuffle();
	boolean hasNext();
	int take(Set<Integer> current, double[] lambda);
	void liked(int v);
	void disliked(int v);
}
