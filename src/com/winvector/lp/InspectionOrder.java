package com.winvector.lp;


public interface InspectionOrder {
	void startPass();
	void shuffle();
	boolean hasNext();
	int take(int[] basis, double[] lambda);
	void liked(int v);
	void disliked(int v);
}
