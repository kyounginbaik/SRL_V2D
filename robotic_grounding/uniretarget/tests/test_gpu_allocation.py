"""Pure CPU allocation tests; never inspect or allocate the host's GPUs."""
import importlib.util
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('uniretarget_gpu_allocation', PACKAGE/'gpu_allocation.py')
gpu = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gpu
spec.loader.exec_module(gpu)


class GPUAllocationTest(unittest.TestCase):
    def setUp(self):
        # XML order deliberately differs from NVIDIA index order.
        xml = '''<nvidia_smi_log>
          <gpu><uuid>GPU-B</uuid><minor_number>0</minor_number></gpu>
          <gpu><uuid>GPU-A</uuid><minor_number>2</minor_number></gpu>
          <gpu><uuid>GPU-C</uuid><minor_number>7</minor_number></gpu>
        </nvidia_smi_log>'''
        self.devices = gpu.parse_inventory(xml, '0, GPU-A\n1, GPU-B\n2, GPU-C\n')

    def resolve(self, **environ):
        return gpu.resolve_gpu(environ, self.devices).uuid

    def test_inventory_joins_by_uuid_not_xml_order(self):
        self.assertEqual(self.devices[0], gpu.GPU(0, 2, 'GPU-A'))

    def test_auto_rejects_different_index_and_minor_matches(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous.*UNIRETARGET_GPU_ID_KIND'):
            self.resolve(SLURM_JOB_GPUS='0')

    def test_explicit_site_convention_resolves_ambiguity(self):
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='0', UNIRETARGET_GPU_ID_KIND='minor'), 'GPU-B')
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='0', UNIRETARGET_GPU_ID_KIND='index'), 'GPU-A')

    def test_auto_accepts_one_unique_match(self):
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='7'), 'GPU-C')
        same = (gpu.GPU(0, 0, 'GPU-A'),)
        self.assertEqual(gpu.resolve_gpu({'SLURM_JOB_GPUS': '0'}, same).uuid, 'GPU-A')

    def test_scheduler_uuid_passes_directly(self):
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='GPU-B'), 'GPU-B')

    def test_step_allocation_precedes_job_then_cuda_fallback(self):
        self.assertEqual(self.resolve(SLURM_STEP_GPUS='GPU-C', SLURM_JOB_GPUS='GPU-B',
                                     CUDA_VISIBLE_DEVICES='0'), 'GPU-C')
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='GPU-B', CUDA_VISIBLE_DEVICES='0'), 'GPU-B')
        self.assertEqual(self.resolve(CUDA_VISIBLE_DEVICES='0', UNIRETARGET_GPU_ID_KIND='minor'), 'GPU-A')

    def test_explicit_uuid_disambiguates_but_cannot_select_unallocated_gpu(self):
        self.assertEqual(self.resolve(SLURM_JOB_GPUS='0', UNIRETARGET_GPU_UUID='GPU-B'), 'GPU-B')
        with self.assertRaisesRegex(ValueError, 'outside the scheduler allocation'):
            self.resolve(SLURM_JOB_GPUS='0', UNIRETARGET_GPU_UUID='GPU-C')
        with self.assertRaisesRegex(ValueError, 'outside the scheduler allocation'):
            self.resolve(SLURM_JOB_GPUS='GPU-A', UNIRETARGET_GPU_UUID='GPU-B')
        with self.assertRaisesRegex(ValueError, 'outside the scheduler allocation'):
            self.resolve(SLURM_JOB_GPUS='0', UNIRETARGET_GPU_ID_KIND='index', UNIRETARGET_GPU_UUID='GPU-B')

    def test_explicit_uuid_without_scheduler_still_requires_real_inventory(self):
        self.assertEqual(self.resolve(UNIRETARGET_GPU_UUID='GPU-C'), 'GPU-C')
        for value in ('0', 'GPU-unknown', 'GPU-A,GPU-B'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(UNIRETARGET_GPU_UUID=value)

    def test_no_allocation_and_bad_conventions_fail(self):
        for environ in ({}, {'SLURM_JOB_GPUS': '99'}, {'SLURM_JOB_GPUS': '0', 'UNIRETARGET_GPU_ID_KIND': 'guess'}):
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                self.resolve(**environ)

    def test_multiple_or_malformed_devices_are_never_silently_truncated(self):
        for value in ('0,1', 'GPU-A,GPU-B', '0-1', 'NoDevFiles', '-1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'exactly one'):
                self.resolve(SLURM_JOB_GPUS=value, UNIRETARGET_GPU_UUID='GPU-A')

    def test_bad_inventory_is_rejected(self):
        xml = '<nvidia_smi_log><gpu><uuid>GPU-A</uuid><minor_number>0</minor_number></gpu></nvidia_smi_log>'
        for csv in ('', '0,GPU-X\n', '0,GPU-A\n1,GPU-A\n', '0,GPU-A,extra\n'):
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                gpu.parse_inventory(xml, csv)


if __name__ == '__main__':
    unittest.main()
