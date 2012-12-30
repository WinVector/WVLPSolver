package com.winvector.lp.impl;


import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

public class TestInspectionOrder {
	@Test
	public void testShift() {
		final Random rand = new Random(3626L);
		final InspectionOrder io = new InspectionOrder(5,rand);
		io.startPass();
		for(int i=0;i<5;++i) {
			assertTrue(io.hasNext());
			io.take();
		}
		assertFalse(io.hasNext());
	}
}
