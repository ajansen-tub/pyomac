import unittest

from pyoma import fdd, peak_picking

from statespace_model import get_model
from simulation import generate_impulse_data, generate_random_data, assert_modal_identification


class FddTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        m, k, c_damp, cls.f_a, cls.ms_a, cls.zeta_a = get_model()
        cls.t_impulse, cls.acc_impulse, cls.fs = generate_impulse_data(m, k, c_damp)
        cls.t_random, cls.acc_random, _ = generate_random_data(m, k, c_damp)

    def test_impulse_welch(self):

        f, s, u = fdd(self.acc_impulse, self.fs, method='welch')
        f_e, ms_e = peak_picking(f, s, u, n_sv=1, thres=0.1)
        assert_modal_identification(self, 'Test impulse with fdd_welch', f_e, ms_e)

    def test_impulse_direct(self):

        f, s, u = fdd(self.acc_impulse, self.fs, method='direct')
        f_e, ms_e = peak_picking(f, s, u, n_sv=1, thres=0.1)
        assert_modal_identification(self, 'Test impulse with fdd_direct', f_e, ms_e)

    def test_random_welch(self):

        f, s, u = fdd(self.acc_random, self.fs, method='welch')
        f_e, ms_e = peak_picking(f, s, u, n_sv=1, thres=0.1)
        assert_modal_identification(self, 'Test random with fdd_welch', f_e, ms_e)

    def test_random_direct(self):

        f, s, u = fdd(self.acc_random, self.fs, method='direct')
        f_e, ms_e = peak_picking(f, s, u, n_sv=1, thres=0.1)
        assert_modal_identification(self, 'Test random with fdd_direct', f_e, ms_e)


# TODO: implement fft and cpsd test cases

if __name__ == '__main__':
    unittest.main()
