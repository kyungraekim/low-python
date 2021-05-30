import unittest
from .ndarray import NumpyIndexingArray


def _get_sample_1darray():
    return [i for i in range(24)]


def _get_sample_3darray():
    return [[[i + 2 * j + 6 * k for i in range(2)] for j in range(3)] for k in range(4)]


def _zip_nested(array, base_array):
    iterators = [iter(arr) for arr in [array, base_array]]
    sentinel = object()
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        if hasattr(result[-1], '__iter__'):
            yield from _zip_nested(*result)
        else:
            yield tuple(result)


class NumpyIndexingArrayTest(unittest.TestCase):
    def testT(self):
        arr = NumpyIndexingArray(_get_sample_3darray()).T
        self.assertEqual(arr.shape, (4, 3, 2))
        for expected, actual in enumerate(arr):
            self.assertEqual(actual, expected)

    def testFlat(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        for expected, actual in enumerate(arr.flat):
            self.assertEqual(expected, actual)

    def testShape(self):
        # arr = NumpyIndexingArray(_get_sample_1darray())
        # self.assertEqual(arr.shape, (24,))
        arr = NumpyIndexingArray(_get_sample_3darray())
        self.assertEqual(arr.shape, (4, 3, 2))

    def testReshape(self):
        arr = NumpyIndexingArray(_get_sample_1darray())
        self.assertEqual(arr.reshape(2, 3, 4).shape, (2, 3, 4))
        self.assertEqual(arr.reshape(4, 6).shape, (4, 6))

    def testValueFromSimpleIndexing(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(arr[i, j, k], arr[i][j][k])
                    self.assertEqual(arr[i, j, k], 6 * i + 2 * j + i)

    def _assert_equal_nested(self, actual_arr, expected_arr):
        for actual, expected in _zip_nested(actual_arr, expected_arr):
            self.assertEqual(expected, actual)

    def testValueFromSliceIndexing(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        self._assert_equal_nested(arr[:, 0, 0], [0, 6, 12, 18])
        self._assert_equal_nested(arr[0, :, 0], [0, 2, 4])
        self._assert_equal_nested(arr[0, 0, :], [0, 1])
        self._assert_equal_nested(arr[:, 0, :], [[0, 1], [6, 7], [12, 13], [18, 19]])
        self._assert_equal_nested(arr[:, :, 0], [[0, 2, 4],
                                                 [6, 8, 10],
                                                 [12, 14, 16],
                                                 [18, 20, 22]])
        self._assert_equal_nested(arr[0, :, :], [[0, 1], [2, 3], [4, 5]])
        self._assert_equal_nested(arr[:, :, 1:4], [[[1], [3], [5]],
                                                   [[7], [9], [11]],
                                                   [[13], [15], [17]],
                                                   [[19], [21], [23]]])
        self._assert_equal_nested(arr[:, 0:3, :], [[[0, 1], [2, 3], [4, 5]],
                                                   [[6, 7], [8, 9], [10, 11]],
                                                   [[12, 13], [14, 15], [16, 17]],
                                                   [[18, 19], [20, 21], [22, 23]]])

    def testValueFromEllipsis(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        self._assert_equal_nested(arr[..., 0, 0], [0, 6, 12, 18])
        self._assert_equal_nested(arr[0, ..., 0], [0, 2, 4])
        self._assert_equal_nested(arr[0, 0, ...], [0, 1])
        self._assert_equal_nested(arr[..., 0, :], [[0, 1], [6, 7], [12, 13], [18, 19]])
        self._assert_equal_nested(arr[..., 0], [[0, 2, 4],
                                                 [6, 8, 10],
                                                 [12, 14, 16],
                                                 [18, 20, 22]])
        self._assert_equal_nested(arr[0, ...], [[0, 1], [2, 3], [4, 5]])
        self._assert_equal_nested(arr[..., 1:4], [[[1], [3], [5]],
                                                   [[7], [9], [11]],
                                                   [[13], [15], [17]],
                                                   [[19], [21], [23]]])
        self._assert_equal_nested(arr[..., 0:3, :], [[[0, 1], [2, 3], [4, 5]],
                                                   [[6, 7], [8, 9], [10, 11]],
                                                   [[12, 13], [14, 15], [16, 17]],
                                                   [[18, 19], [20, 21], [22, 23]]])

    def testAssignSingleValue(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        answer = _get_sample_3darray()
        arr[0, 0, 0] = answer[0][0][0] = 100
        arr[1][0][1] = answer[1][0][1] = 200
        arr[2, 1][0] = answer[2][1][0] = 300
        arr[3][1, 1] = answer[3][1][1] = 400
        self._assert_equal_nested(arr, answer)

    def testAssignSlice(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        answer = _get_sample_3darray()
        arr[:, 0, 0] = 100
        for matrix in answer:
            matrix[0][0] = 100
        self._assert_equal_nested(arr, answer)
        arr[0, :, 1] = 200
        for vector in answer[0]:
            vector[1] = 200
        self._assert_equal_nested(arr, answer)
        arr[1, 1, :] = 300
        for i in range(len(answer[1][1])):
            answer[1][1][i] = 300
        self._assert_equal_nested(arr, answer)
        arr[:, :, 0] = 400
        for matrix in answer:
            for vector in matrix:
                vector[0] = 400
        self._assert_equal_nested(arr, answer)
        arr[:, 2, :] = 500
        for matrix in answer:
            for i in range(len(matrix[2])):
                matrix[2][i] = 500
        arr[3, :, :] = 600
        for vector in answer[3]:
            for i in range(len(vector)):
                vector[i] = 600
        self._assert_equal_nested(arr, answer)
        arr[:, :, :] = 700
        for matrix in answer:
            for vector in matrix:
                for i in range(len(vector)):
                    vector[i] = 700
        self._assert_equal_nested(arr, answer)

    def testAssignEllipsis(self):
        arr = NumpyIndexingArray(_get_sample_3darray())
        answer = _get_sample_3darray()
        arr[..., 0, 0] = 100
        for matrix in answer:
            matrix[0][0] = 100
        self._assert_equal_nested(arr, answer)
        arr[0, ..., 1] = 200
        for vector in answer[0]:
            vector[1] = 200
        self._assert_equal_nested(arr, answer)
        arr[1, 1, ...] = 300
        for i in range(len(answer[1][1])):
            answer[1][1][i] = 300
        self._assert_equal_nested(arr, answer)
        arr[..., 0] = 400
        for matrix in answer:
            for vector in matrix:
                vector[0] = 400
        self._assert_equal_nested(arr, answer)
        arr[..., 2, :] = 500
        for matrix in answer:
            for i in range(len(matrix[2])):
                matrix[2][i] = 500
        arr[3, ...] = 600
        for vector in answer[3]:
            for i in range(len(vector)):
                vector[i] = 600
        self._assert_equal_nested(arr, answer)
        arr[...] = 700
        for matrix in answer:
            for vector in matrix:
                for i in range(len(vector)):
                    vector[i] = 700
        self._assert_equal_nested(arr, answer)

    def testToList(self):
        arr = NumpyIndexingArray(_get_sample_1darray()).tolist()
        self._assert_equal_nested(arr, _get_sample_1darray())
        arr = NumpyIndexingArray(_get_sample_3darray()).tolist()
        self._assert_equal_nested(arr, _get_sample_3darray())

    def testSum(self):
        ref = _get_sample_3darray()
        arr = NumpyIndexingArray(_get_sample_3darray())
        answer = 0
        for matrix in ref:
            for vector in matrix:
                for value in vector:
                    answer += value
        self.assertEqual(answer, arr.sum())

        axis = 0
        actual = arr.sum(axis=axis)
        answer = []
        for vector in ref[0]:
            ax2_sum = []
            for _ in vector:
                ax2_sum.append(0)
            answer.append(ax2_sum)
        for matrix in ref:
            for ax1, vector in enumerate(matrix):
                for ax2, value in enumerate(vector):
                    answer[ax1][ax2] += value
        self._assert_equal_nested(actual, answer)

        axis = 1
        actual = arr.sum(axis=axis)
        answer = []
        for vector in ref:
            ax2_sum = []
            for _ in vector[0]:
                ax2_sum.append(0)
            answer.append(ax2_sum)
        for ax0, matrix in enumerate(ref):
            for vector in matrix:
                for ax2, value in enumerate(vector):
                    answer[ax0][ax2] += value
        self._assert_equal_nested(actual, answer)

        axis = (1, 2)
        actual = arr.sum(axis=axis)
        answer = [0 for _ in range(len(ref))]
        for ax0, matrix in enumerate(ref):
            for ax1, vector in enumerate(matrix):
                for value in vector:
                    answer[ax0] += value
        self._assert_equal_nested(actual, answer)
