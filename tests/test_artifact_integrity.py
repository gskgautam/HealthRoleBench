import csv
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.reporting.report_tables import verify_encoder_predictions


def write_rows(path, header, data):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(data)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_failed_write_is_unpublished_and_existing_file_is_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'private.csv'
            with self.assertRaises(RuntimeError):
                with private_text_artifact(path) as stream:
                    stream.write('partial data')
                    raise RuntimeError('simulated interruption')
            self.assertFalse(path.exists())
            with private_text_artifact(path) as stream:
                stream.write('complete data')
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                with private_text_artifact(path):
                    self.fail('existing artifact must remain intact')
            self.assertEqual(path.read_text(), 'complete data')

    def test_metrics_recompute_and_tampering_is_rejected(self):
        labels = ['a', 'b', 'c', 'd', 'e']
        matrix = [[int(i == j) for j in range(5)] for i in range(5)]
        metrics = {'labels': labels, 'train_n': 1, 'validation_n': 1, 'test_n': 5,
                   'confusion_matrix': matrix, 'classification_report': {'macro avg': {'f1-score': 1}}}
        with TemporaryDirectory() as directory:
            path = Path(directory)
            split = [('train', 'g', 'train', '0'), ('validation', 'g', 'val', '1')]
            split.extend(('test', 'g', label, '2') for label in labels)
            split_header = ('partition', 'generator', 'sample_id', 'group')
            write_rows(path/'split.csv', split_header, split)
            pred_header = ('generator', 'sample_id', 'label', 'prediction')
            predictions = [('g', label, label, label) for label in labels]
            write_rows(path/'predictions.csv', pred_header, predictions)
            self.assertEqual(verify_encoder_predictions(path, metrics), 1)
            write_rows(path/'predictions.csv', pred_header, predictions[:-1] + predictions[:1])
            with self.assertRaisesRegex(ValueError, 'Repeated or non-test'):
                verify_encoder_predictions(path, metrics)
            split[0] = ('train', 'g', 'train', '2')
            write_rows(path/'split.csv', split_header, split)
            with self.assertRaisesRegex(ValueError, 'crosses partitions'):
                verify_encoder_predictions(path, metrics)
