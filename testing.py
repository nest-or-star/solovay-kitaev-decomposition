import unittest
import numpy as np
from scipy.linalg import expm

from quantum_decomposition import (
    remove_global_phase,
    add_global_phase,
    extract_axis_angle,
    rotation_matrix,
    solve_unitary_conjugate,
    gc_decomposition,
    compare_su2,
    is_unitary,
    align_phase,
    X, Y, Z, I
)

class TestQuantumDecompositionMath(unittest.TestCase):

    def setUp(self):
        self.H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self.S = np.array([[1, 0], [0, 1j]], dtype=complex)
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        theta = np.pi / 3
        self.random_su2 = rotation_matrix(axis, theta)

    def test_is_unitary(self):
        self.assertTrue(is_unitary(self.H))
        self.assertTrue(is_unitary(X))
        
        non_unitary = np.array([[1, 2], [3, 4]], dtype=complex)
        self.assertFalse(is_unitary(non_unitary))

    def test_remove_global_phase(self):
        U = 1j * X
        su2_matrix, phase = remove_global_phase(U)
        det = np.linalg.det(su2_matrix)
        self.assertAlmostEqual(det, 1.0)
        
        reconstructed = su2_matrix * np.exp(1j * phase)
        self.assertTrue(np.allclose(U, reconstructed))

        U = np.exp(1j * np.pi / 6) * self.random_su2
        su2_matrix, phase = remove_global_phase(U)
        det = np.linalg.det(su2_matrix)
        self.assertAlmostEqual(det, 1.0)

        reconstructed = su2_matrix * np.exp(1j * phase)
        self.assertTrue(np.allclose(U, reconstructed))

    def test_rotation_matrix_and_extract_axis_angle(self):
        test_axis = np.array([0, 1, 0], dtype=float) # Y-axis
        test_theta = np.pi / 2
        
        R = rotation_matrix(test_axis, test_theta)
        
        extracted_axis, extracted_theta = extract_axis_angle(R)
        
        self.assertAlmostEqual(test_theta, extracted_theta)
        
        self.assertTrue(np.allclose(test_axis, extracted_axis) or 
                        np.allclose(test_axis, -extracted_axis))

    def test_solve_unitary_conjugate(self):
        S_expected = rotation_matrix(np.array([1, 0, 0]), np.pi/4)
        W = Z
        V = S_expected @ W @ S_expected.conj().T
        
        S_calculated = solve_unitary_conjugate(V, W)
        
        V_reconstructed = S_calculated @ W @ S_calculated.conj().T
        
        self.assertTrue(np.allclose(V, V_reconstructed, atol=1e-10))

    def test_gc_decomposition(self):
        
        target_axis = np.array([0, 0, 1])
        target_theta = 0.5  # Small angle
        U_target = rotation_matrix(target_axis, target_theta)
        
        A, B = gc_decomposition(U_target)
        
        commutator = A @ B @ A.conj().T @ B.conj().T
        
        dist = compare_su2(U_target, commutator)
        
        self.assertLess(dist, 1e-10)

    def test_compare_su2(self):
        self.assertAlmostEqual(compare_su2(X, X), 0.0)
        
        self.assertAlmostEqual(compare_su2(X, 1j * X), 0.0)
        
        dist = compare_su2(X, Z)
        self.assertGreater(dist, 0.1)

    def test_align_phase(self):
        target = I 
        source = 1j * I
        
        aligned = align_phase(source, target)
        
        self.assertTrue(np.allclose(aligned, target))

    def test_add_global_phase(self):
        phase = np.pi / 2
        U = I
        U_shifted = add_global_phase(U, phase)
        expected = 1j * I
        self.assertTrue(np.allclose(U_shifted, expected))

if __name__ == '__main__':
    unittest.main()