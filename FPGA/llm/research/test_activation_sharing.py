import unittest

from activation_sharing_experiment import (
    Config, execute, experiment, find_activations, fixture,
)


class ActivationSharingTests(unittest.TestCase):
    def test_signed_threshold_and_tails(self):
        x = [-128, -1, 0, 1, 127] * 13
        for shift in range(16):
            for bw in (8, 64):
                indices, cycles = find_activations(x, shift, Config(activation_bw=bw))
                expected = [i for i, v in enumerate(x) if abs(v) >= 2**shift]
                self.assertEqual(indices, expected)
                self.assertGreaterEqual(cycles, (len(x) + bw - 1) // bw)

    def test_single_row_has_no_reuse_gain(self):
        x, w = fixture(137, 1, .1)
        out = {v['mode']: v for v in experiment(x, w, 0, Config())}
        self.assertEqual(out['shared_replay']['cycles'], out['repeated_finder']['cycles'])

    def test_sharing_changes_only_preparation(self):
        x, w = fixture(129, 11, .5, weight_density=.5, w4=True)
        for layout in ('row_major', 'output_interleaved'):
            out = {v['mode']: v for v in experiment(x, w, 2, Config(layout=layout))}
            a, b = out['repeated_finder'], out['shared_replay']
            for key in ('weight_bytes', 'compute_cycles', 'useful_macs', 'issued_macs', 'descriptor_reads'):
                self.assertEqual(a[key], b[key], key)
            self.assertEqual(a['activation_source_bytes'], 11 * b['activation_source_bytes'])
            self.assertEqual(a['frontend_cycles'], 11 * b['frontend_cycles'])
            self.assertEqual(out['shared_broadcast']['descriptor_reads'], 2 * b['kept'])

    def test_bank_conflict_has_real_cost(self):
        x = [1] * 64
        w = [[1] * 64]
        free = execute(x, w, list(range(8)), Config())
        conflict = execute(x, w, list(range(0, 64, 8)), Config())
        self.assertEqual(free['result'], [8])
        self.assertEqual(conflict['result'], [8])
        self.assertEqual(conflict['compute_cycles'] - free['compute_cycles'], 7)

    def test_latency_and_response_backpressure(self):
        x = [1] * 128
        w = [[2] * 128 for _ in range(8)]
        fast = execute(x, w, list(range(128)), Config())
        slow = execute(x, w, list(range(128)), Config(memory_latency=9, response_capacity=8, window=7))
        self.assertEqual(slow['result'], fast['result'])
        self.assertEqual(slow['weight_bytes'], 1024)
        self.assertGreater(slow['compute_cycles'], fast['compute_cycles'])

    def test_empty_and_zero_weights(self):
        for x, w in [([], [[] for _ in range(9)]), ([0] * 65, [[1] * 65]),
                     ([1] * 65, [[0] * 65])]:
            out = experiment(x, w, 0, Config())
            self.assertTrue(all(r['useful_macs'] == 0 for r in out))
            self.assertTrue(all(r['cycles'] > 0 for r in out))
        self.assertEqual(out[2]['weight_bytes'], 65)  # Cannot skip unread zero weights.

    def test_int32_overflow(self):
        x = [-128] * 131080
        result = execute(x, [x], list(range(len(x))), Config())
        self.assertEqual(result['result'], [-2147352576])  # 0x80020000


if __name__ == '__main__':
    unittest.main()
