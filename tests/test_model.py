import json
import unittest
import numpy as np
from common import ROOT
from model import predict,size,depth,replace,subtree,random_tree,variation,valid,generate,run

class TestGP(unittest.TestCase):
    def test_arithmetic_and_protection(self):
        x=np.array([-2.,0.,2.]); t=('+',('*','x','x'),1.)
        np.testing.assert_array_equal(predict(t,x),[5,1,5]); self.assertEqual(size(t),5); self.assertEqual(depth(t),3)
        np.testing.assert_array_equal(predict(('/','x',0.),x),np.ones(3))
    def test_tree_replacement(self):
        t=('+','x',1.); changed=replace(t,(2,),('*','x','x'))
        self.assertEqual(t,('+','x',1.)); self.assertEqual(subtree(changed,(2,)),('*','x','x'))
    def test_operators_preserve_limits_and_finite_output(self):
        cfg=json.loads((ROOT/'config.json').read_text()); rng=np.random.default_rng(1); a=random_tree(rng)
        for _ in range(200):
            a=variation(a,random_tree(rng),rng,cfg)
            self.assertTrue(valid(a,cfg)); self.assertTrue(np.isfinite(predict(a,np.linspace(-2,2,31))).all())
    def test_reproducibility_and_budget(self):
        cfg=json.loads((ROOT/'config.json').read_text()); cfg['generations']=2
        data=generate(cfg['data_seed'])
        splits={s:(np.array([r['x'] for r in data if r['split']==s]),np.array([r['y'] for r in data if r['split']==s])) for s in ['train','validation','test']}
        self.assertEqual([len(splits[s][0]) for s in splits],[120,60,60])
        a=run(cfg,1,splits['train'],splits['validation'],0); b=run(cfg,1,splits['train'],splits['validation'],0)
        self.assertEqual(a[0],b[0]); self.assertEqual(a[3],3*cfg['population']); self.assertEqual(a[4],cfg['population'])
if __name__=='__main__': unittest.main()
