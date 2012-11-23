package com.winvector.comb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

public class TestAssignment {
	@Test
	public void testAssignment() {
		final double[][] c = {
				{ 1, 10, Double.NaN },
				{ 1, Double.NaN, 4 },
				{ 3, 100, 3 }
		};
		final int[] assignment = Assignment.computeAssignment(c,1000);
		assertNotNull(assignment);
		final boolean valid = Assignment.checkValid(c,assignment);
		assertTrue(valid);
		final int[] expect = {1, 0, 2};
		assertEquals(expect.length,assignment.length);
		for(int i=0;i<expect.length;++i) {
			assertEquals(expect[i],assignment[i]);
		}
	}

	@Test
	public void testRandAssignment() {
		final int n = 20;
		final Random rand = new Random(235135L);
		final double[][] c = new double[n][n];
		for(int i=0;i<n;++i) {
			for(int j=0;j<n;++j) {
				c[i][j] = rand.nextDouble();
			}
		}
		final int[] assignment = Assignment.computeAssignment(c,1000);
		assertNotNull(assignment);
		final boolean valid = Assignment.checkValid(c,assignment);
		assertTrue(valid);
	}
}
