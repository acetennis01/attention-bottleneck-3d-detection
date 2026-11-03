"""Integrity checks for compact public research reports; no GPU is required."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublishedResultsTests(unittest.TestCase):
    def test_all_published_reports_match_their_manifest(self):
        manifest = json.loads((ROOT/'results/provenance.json').read_text())
        self.assertEqual(manifest['completed_camera_cases'], 30)
        for record in manifest['files']:
            path = ROOT/'results'/record['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             record['sha256'], record['path'])

    def test_camera_suite_is_complete_and_checkpoint_consistent(self):
        suite = json.loads((ROOT/'results/camera_ablations/suite.json').read_text())
        self.assertEqual(suite['status'], 'complete')
        self.assertEqual(len(suite['cases']), 30)
        self.assertTrue(all(x['status'] == 'passed' for x in suite['cases'].values()))
        for model in suite['models']:
            config = ROOT/'results/saved_configs'/(model['name']+'.py')
            self.assertEqual(hashlib.sha256(config.read_bytes()).hexdigest(), model['config_sha256'])
            for mode in ['correct', 'black', 'shuffled']:
                report = json.loads((ROOT/'results/camera_ablations'/model['name']/(mode+'.json')).read_text())
                self.assertEqual(report['checkpoint_sha256'], model['checkpoint_sha256'])
                self.assertEqual(report['validation_samples'], model['validation_samples'])
                self.assertTrue(report['strict_checkpoint_load_passed'])

    def test_strong_model_uses_selected_best_nds_checkpoint(self):
        path = ROOT/'results/camera_ablations/nuscenes_transfusion_swint_b8/correct_devkit.json'
        report = json.loads(path.read_text())
        self.assertAlmostEqual(report['mean_ap']*100, 64.79498294958124)
        self.assertAlmostEqual(report['nd_score']*100, 69.3897358537647)
        self.assertEqual(f"{report['mean_ap']*100:.2f}", '64.79')

    def test_selected_figures_match_sources(self):
        directory = ROOT/'docs/figures/kitti_v2'
        manifest = json.loads((directory/'provenance.json').read_text())
        for record in manifest['files']:
            self.assertEqual(hashlib.sha256((directory/record['path']).read_bytes()).hexdigest(),
                             record['sha256'])

    def test_no_large_or_raw_predictions_in_published_results(self):
        for path in (ROOT/'results').rglob('*'):
            if not path.is_file():
                continue
            self.assertNotIn(path.suffix, ['.pth', '.pkl', '.npz', '.mp4'])
            self.assertNotEqual(path.name, 'results_nusc.json')
            self.assertLess(path.stat().st_size, 1024*1024)


if __name__ == '__main__':
    unittest.main()
