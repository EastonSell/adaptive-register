import itertools, unittest
from controls import build,score,route_known_complement
from analyze import analyze,DEFAULT,IDENTITY
import copy
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.prompts,cls.targets,cls.renders,cls.summary=build()
    def test_input_only_router(self):
        for p,t in zip(self.prompts,self.targets):
            self.assertTrue(score(route_known_complement(p['model_input']),t['target_ids'])['exact_pair'])
            swapped=copy.deepcopy(p['model_input'])
            swapped['reader_knows']=[c['term'] for c in swapped['candidates'] if c['term'] not in p['model_input']['reader_knows']]
            self.assertEqual(score(route_known_complement(swapped),t['target_ids'])['unknown_recall'],0)
    def test_annotated_denominators(self):
        result,_=analyze(DEFAULT,IDENTITY)
        self.assertEqual(result['proposition_judgment_count'],74)
        self.assertEqual(result['preserved_proposition_judgments'],64)
        self.assertEqual(result['failed_proposition_judgments'],10)
    def test_pairs_by_source_not_row_order(self):
        groups={}
        for p in reversed(self.prompts):
            key=(p['model_input']['source'],tuple(c['term'] for c in p['model_input']['candidates']))
            groups.setdefault(key,[]).append(p['model_input'])
        self.assertEqual(len(groups),288)
        for pair in groups.values():
            self.assertEqual(len(pair),2)
            self.assertEqual(set(pair[0]['reader_knows'])|set(pair[1]['reader_knows']),{c['term'] for c in pair[0]['candidates']})
            self.assertFalse(set(pair[0]['reader_knows'])&set(pair[1]['reader_knows']))
    def test_counts(self):
        self.assertEqual(len(self.prompts),576);self.assertEqual(len(self.renders),24)
    def test_oracle_and_blind_null(self):
        self.assertEqual(self.summary['oracle_exact_pairs'],576)
        self.assertEqual(self.summary['fixed_candidate_01_recall'],0.5)
        self.assertEqual(self.summary['swapped_oracle_recall'],0)
    def test_bad_outputs_fail_closed(self):
        for x in [None,[], '0,0','0,1,2,3','1,0','0,4','0,1\n','0 1',' 0,1','٠,١']:
            self.assertFalse(score(x,[0,1])['valid'])
    def test_pair_causal_identity(self):
        for offset in range(0,576,48):
            for perm in range(24):
                a,b=self.prompts[offset+perm]['model_input'],self.prompts[offset+24+perm]['model_input']
                self.assertEqual(a['source'],b['source']);self.assertEqual(a['candidates'],b['candidates'])
                self.assertFalse(set(a['reader_knows'])&set(b['reader_knows']))
                ta,tb=self.targets[offset+perm]['target_ids'],self.targets[offset+24+perm]['target_ids']
                for pair in itertools.combinations(range(4),2):
                    response=','.join(map(str,pair))
                    self.assertEqual(score(response,ta)['unknown_recall']+score(response,tb)['unknown_recall'],1)
                    self.assertEqual(score(response,ta)['selectivity']+score(response,tb)['selectivity'],0)
                    self.assertLessEqual(int(score(response,ta)['exact_pair'])+int(score(response,tb)['exact_pair']),1)
    def test_no_evaluation_answers_or_unknown_labels_in_input(self):
        allowed={'instruction','source','reader_knows','candidates'}
        for p in self.prompts: self.assertEqual(set(p['model_input']),allowed)
    def test_scaffold_source_is_exact(self):
        lookup={p['case_id']:p for p in self.prompts}
        for r in self.renders:self.assertEqual(r['source_prefix'],lookup[r['case_id']]['model_input']['source'])
    def test_candidate_definitions_copied_between_profiles(self):
        for offset in range(0,576,48):
            self.assertEqual(self.prompts[offset]['model_input']['candidates'],self.prompts[offset+24]['model_input']['candidates'])
    def test_invalid_is_not_negative_oracle(self):
        self.assertEqual(score('all',[0,1])['selectivity'],0)
        self.assertEqual(score('2,3',[0,1])['selectivity'],-1)
if __name__=='__main__': unittest.main(verbosity=2)
