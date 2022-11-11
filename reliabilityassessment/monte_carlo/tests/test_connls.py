from pathlib import Path

import numpy as np
from numpy import loadtxt

from reliabilityassessment.monte_carlo.connls import connls


def test_connls():

    NX, NR, NLS = 4, 1, 1
    # NL = 5 # not used
    NR -= 1  # 0-based index in Python!

    TEST_DIR = Path(__file__).parent.absolute()

    # -----------------------Test case 1 ----------------------
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/BC")
    BC = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/INJ")
    INJ = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/INJB")
    INJB = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/LT")
    LT = loadtxt(FileNameAndPath).astype(int)
    LT -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/BLP")
    BLP = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/LP")
    LP = loadtxt(FileNameAndPath).astype(int)
    LP -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/BN")
    BN = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/LOD")
    LOD = loadtxt(FileNameAndPath)

    M, N, N1, A, XOB, XOBI, IBAS, BS, B, B1, TAB = connls(
        BC, INJ, INJB, NX, NR, LT, BLP, LP, BN, LOD, NLS
    )

    M_true, N_true, N1_true = 30, 43, 38
    assert M == M_true
    assert N == N_true
    assert N1 == N1_true

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/A")
    A_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(A, A_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/XOB")
    XOB_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOB, XOB_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/XOBI")
    XOBI_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOBI, XOBI_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/IBAS")
    IBAS_true = loadtxt(FileNameAndPath).astype(int)
    IBAS_true -= 1  # 0-based index in Python!
    IBAS_true = IBAS_true[:M]
    np.testing.assert_array_equal(IBAS, IBAS_true)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/BS")
    BS_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(BS, BS_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/B")
    B_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(B, B_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/B1")
    B1_true = loadtxt(FileNameAndPath)
    B1_true = B1_true[:M]
    np.testing.assert_array_almost_equal(B1, B1_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case1/TAB")
    TAB_true = loadtxt(FileNameAndPath)
    TAB_true = TAB_true[:M, :N]
    np.testing.assert_array_almost_equal(TAB, TAB_true, decimal=5)

    # -----------------------Test case 2 ----------------------
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/BC")
    BC = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/INJ")
    INJ = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/INJB")
    INJB = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/LT")
    LT = loadtxt(FileNameAndPath).astype(int)
    LT -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/BLP")
    BLP = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/LP")
    LP = loadtxt(FileNameAndPath).astype(int)
    LP -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/BN")
    BN = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/LOD")
    LOD = loadtxt(FileNameAndPath)

    M, N, N1, A, XOB, XOBI, IBAS, BS, B, B1, TAB = connls(
        BC, INJ, INJB, NX, NR, LT, BLP, LP, BN, LOD, NLS
    )

    M_true, N_true, N1_true = 30, 43, 38
    assert M == M_true
    assert N == N_true
    assert N1 == N1_true

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/A")
    A_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(A, A_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/XOB")
    XOB_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOB, XOB_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/XOBI")
    XOBI_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOBI, XOBI_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/IBAS")
    IBAS_true = loadtxt(FileNameAndPath).astype(int)
    IBAS_true -= 1  # 0-based index in Python!
    IBAS_true = IBAS_true[:M]
    np.testing.assert_array_equal(IBAS, IBAS_true)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/BS")
    BS_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(BS, BS_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/B")
    B_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(B, B_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/B1")
    B1_true = loadtxt(FileNameAndPath)
    B1_true = B1_true[:M]
    np.testing.assert_array_almost_equal(B1, B1_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case2/TAB")
    TAB_true = loadtxt(FileNameAndPath)
    TAB_true = TAB_true[:M, :N]
    np.testing.assert_array_almost_equal(TAB, TAB_true, decimal=5)

    # -----------------------Test case 3 ----------------------
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/BC")
    BC = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/INJ")
    INJ = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/INJB")
    INJB = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/LT")
    LT = loadtxt(FileNameAndPath).astype(int)
    LT -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/BLP")
    BLP = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/LP")
    LP = loadtxt(FileNameAndPath).astype(int)
    LP -= 1  # 0-based index in Python!
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/BN")
    BN = loadtxt(FileNameAndPath)
    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/LOD")
    LOD = loadtxt(FileNameAndPath)

    M, N, N1, A, XOB, XOBI, IBAS, BS, B, B1, TAB = connls(
        BC, INJ, INJB, NX, NR, LT, BLP, LP, BN, LOD, NLS
    )

    M_true, N_true, N1_true = 30, 43, 38
    assert M == M_true
    assert N == N_true
    assert N1 == N1_true

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/A")
    A_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(A, A_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/XOB")
    XOB_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOB, XOB_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/XOBI")
    XOBI_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(XOBI, XOBI_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/IBAS")
    IBAS_true = loadtxt(FileNameAndPath).astype(int)
    IBAS_true -= 1  # 0-based index in Python!
    IBAS_true = IBAS_true[:M]
    np.testing.assert_array_equal(IBAS, IBAS_true)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/BS")
    BS_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(BS, BS_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/B")
    B_true = loadtxt(FileNameAndPath)
    np.testing.assert_array_almost_equal(B, B_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/B1")
    B1_true = loadtxt(FileNameAndPath)
    B1_true = B1_true[:M]
    np.testing.assert_array_almost_equal(B1, B1_true, decimal=5)

    FileNameAndPath = Path(TEST_DIR, "testdata_connls/case3/TAB")
    TAB_true = loadtxt(FileNameAndPath)
    TAB_true = TAB_true[:M, :N]
    np.testing.assert_array_almost_equal(TAB, TAB_true, decimal=5)

    return
