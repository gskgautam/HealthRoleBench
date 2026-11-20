import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from healthrolebench.curation.definition_tables import TAXONOMY
from healthrolebench.curation.human_audit import ROLES, labels_from_row, score
from healthrolebench.reporting.stress_tables import decode, masker


def write(path, header, data):
    with path.open('w',newline='',encoding='utf-8') as stream:
        out = csv.writer(stream)
        out.writerow(header)
        out.writerows(data)


class PaperTablesTests(unittest.TestCase):
    def test_masking_matches_whole_words_spaces_and_apostrophes(self):
        with TemporaryDirectory() as directory:
            cfg = Path(directory)/'protocol.json'
            definitions = {role:['viral post',"don't know"] for role,_,_ in TAXONOMY}
            cfg.write_text(json.dumps({'F3':{'lexicons':definitions}}))
            pattern = masker(cfg)
            text = 'A VIRAL   post and don’t know; antiviral post.'
            self.assertEqual(pattern.sub('[MASKED]',text),'A [MASKED] and [MASKED]; antiviral post.')
        self.assertEqual(decode('Fear Monger',ROLES),'Fear Monger')
        self.assertEqual(decode('',ROLES),'__invalid__')

    def test_audit_allows_unsure_and_rejects_mixed_otherwise(self):
        self.assertEqual(labels_from_row({'audit_id':'x','primary_role':'UNSURE','secondary_roles':''}),('UNSURE',()))
        with self.assertRaises(ValueError):
            labels_from_row({'audit_id':'x','primary_role':'Otherwise','secondary_roles':'Fear Monger'})

    def test_blind_ratings_and_multilabel_metrics_are_reproducible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root/'data/experiments/corpus_tables_v1'
            expected.mkdir(parents=True)
            ids = list(zip(map(str,range(len(ROLES))),ROLES))
            write(expected/'audit_key_private.csv',('audit_id','intended_role'),ids)
            header = ('audit_id','primary_role','secondary_roles')
            ratings = [(i,role,'') for i,role in ids]
            for name in ('a','b','consensus','prediction'):
                write(root/f'{name}.csv',header,ratings)
            score(root,root/'a.csv',root/'b.csv',root/'consensus.csv',None)
            score(root,root/'a.csv',root/'b.csv',root/'consensus.csv',[('primary',str(root/'prediction.csv'))])
            report = root/'data/experiments/human_audit_v1/table_17_multilabel_evaluation.csv'
            with report.open() as stream:
                metrics = next(csv.DictReader(stream))
            self.assertEqual(float(metrics['macro_f1']),1)
            self.assertEqual(float(metrics['hamming_loss']),0)
