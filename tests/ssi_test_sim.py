import unittest
from pyoma import ssi_cov

from statespace_model import get_model
from simulation import generate_random_data, assert_modal_identification


class SsiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m, k, c_damp, cls.f_a, cls.ms_a, cls.zeta_a = get_model()
        cls.t_random, cls.acc_random, cls.fs = generate_random_data(m, k, c_damp)

    def test_random(self):

        f_e, ms_e, zeta_e = ssi_cov(self.acc_random, self.fs)
        assert_modal_identification(self, 'Test random with ssi-cov', f_e, ms_e, zeta_e)


if __name__ == '__main__':
    unittest.main()
