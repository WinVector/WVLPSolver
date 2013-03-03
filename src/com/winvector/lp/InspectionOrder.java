package com.winvector.lp;

public interface InspectionOrder {
	void startPass();
	void shuffle();
	boolean hasNext();
	int take();
	void liked(int v);
	void disliked(int v);
}
