"""Scientific invariants for the bounded completion analysis."""
import unittest
import numpy as np
from .completion_study import permutations, ordered_sequence
from .coefficient_innovation import CoefficientInnovationModel
from .data import simulate


class CompletionTests(unittest.TestCase):
    def test_permutation_preserves_rows_gaps_and_split_recomputes_lag(self):
        days=np.array([0,2,2,2,5,5,5]); seconds=np.array([0,1.8,.04,.03,2.9,.02,.01])*86400
        rows=[dict(timestamp=d,elapsed_seconds=s,y=r) for d,s,r in zip(days,seconds,[0,1,0,1,1,0,1])]
        a,b=permutations(days,seconds)
        seq=dict(user=0,item=0,r=np.array([v['y'] for v in rows]))
        result=ordered_sequence(seq,rows,4,b)
        np.testing.assert_array_equal(result['r'],seq['r'][b])
        np.testing.assert_array_equal(result['x'],np.r_[0,result['r'][:-1]])
        np.testing.assert_array_equal(result['dt'][1:],seconds[b][1:]/86400)
        self.assertEqual(set(b[:4]),set(range(4)))
        self.assertFalse(np.array_equal(a,b))
        crossing=b.copy();crossing[[2,4]]=crossing[[4,2]]
        with self.assertRaises(ValueError):ordered_sequence(seq,rows,4,crossing)
        with self.assertRaises(ValueError):ordered_sequence(seq,rows,4,np.zeros(7,int))

    def test_model_prior_conditional_shape_and_scale_are_changed_explicitly(self):
        seqs,_,_=simulate(seed=52,users=2,items=2,events=8)
        class Capture:
            def gamma(self,shape,scale):self.args=shape,scale;return 2.
        model=CoefficientInnovationModel(seqs,sampler='joint_blocked',interweave=True)
        capture=Capture(); model.rng=capture
        for scale in (.1,1.,4.):
            model.tau2_prior=(2.,scale)
            value=model._inv_gamma_tau2(np.array([1.,2.,3.]),1.)
            self.assertEqual(value,.5)
            self.assertEqual(capture.args,(3.5,1/(scale+2.5)))

    def test_oracle_matches_independent_prior_likelihood_density_ratios(self):
        from scipy.stats import invgamma
        from .completion_report import hierarchy_oracle
        slopes=np.array([.4,.7,1.]);g0=.7
        for scale in (.1,1.,4.):
            result=hierarchy_oracle(slopes,g0,scale)
            def independent(v):
                return invgamma.logpdf(v,a=2,scale=scale)-len(slopes)*np.log(v)/2-np.sum((slopes-g0)**2)/(2*v)
            for v in (.02,.1,1.,5.):
                direct=independent(v)-independent(.5)
                conditional=invgamma.logpdf(v,a=result['shape'],scale=result['scale'])-invgamma.logpdf(.5,a=result['shape'],scale=result['scale'])
                self.assertAlmostEqual(direct,conditional,places=10)

    def test_report_refuses_incomplete_and_existing_outputs(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .completion_report import curate
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            (source/'status.json').write_text('{"state":"running"}')
            with self.assertRaises(ValueError):curate(source,root/'new',root)
            self.assertFalse((root/'new').exists())
            out=root/'keep';out.mkdir();(out/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):curate(source,out,root)
            self.assertEqual((out/'sentinel').read_text(),'keep')

    def test_completion_refuses_overwrite(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .completion_study import execute
        with TemporaryDirectory() as folder:
            out=Path(folder); sentinel=out/'keep';sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):execute(out,Path('/missing'))
            self.assertEqual(sentinel.read_text(),'keep')


if __name__=='__main__':unittest.main()
