"""Transformation and conditional-target checks for the experimental OU kernel."""
import unittest
import numpy as np
from .data import simulate
from .innovation import InnovationModel
from .investigate import nested_simulate, nested_summaries
from .analysis import scalar_draws


class InnovationTests(unittest.TestCase):
    def setUp(self):
        self.seqs,_,_=simulate(seed=7,users=2,items=2,events=8)
        self.model=InnovationModel(self.seqs,seed=81,sampler='joint_blocked',interweave=True)

    def test_adjacent_gap_check_is_not_a_whole_history_integrity_check(self):
        from .provenance_audit import timing_residual_range
        # Every adjacent interval differs by <1 day, but total drift is impossible
        # under a single unknown first-review time of day.
        days=np.array([0,1,2,3])
        seconds=np.array([0,.4,.4,.4])*86400
        self.assertTrue(np.all(np.abs(np.diff(days)-seconds[1:]/86400)<1))
        self.assertGreater(timing_residual_range(days,seconds),1)
        self.assertLess(timing_residual_range([0,0,1],[0,100,86400]),1)

    def test_feasible_gap_order_does_not_authenticate_same_day_outcomes(self):
        from .provenance_audit import timing_residual_range
        days=np.array([0,2,2,2])
        # The long gap is misplaced among same-day events in the stored order.
        stored=np.array([0,.03,1.8,.04])*86400
        ordered=np.array([0,1.8,.04,.03])*86400
        other=np.array([0,1.8,.03,.04])*86400
        self.assertGreater(timing_residual_range(days,stored),1)
        self.assertLess(timing_residual_range(days,ordered),1)
        self.assertLess(timing_residual_range(days,other),1)
        # The final two recall outcomes can exchange order in feasible paths;
        # elapsed-time feasibility therefore cannot authenticate their lag order.
        r=np.array([0,1,0,1]);alternative=r[[0,1,3,2]]
        self.assertFalse(np.array_equal(r,alternative))

    def test_roundtrip_and_triangular_jacobian(self):
        m=self.model
        # Include very short/long Anki timing and unequal history lengths.
        m.DT[0,1:3]=[1/86400,2000]
        z=m.rng.normal(size=m.psi.shape)
        psi=m.reconstruct(z)
        np.testing.assert_allclose(m.innovations(psi),z,atol=1e-10)
        zflat=z[m.M];eps=1e-5
        jac=np.empty((len(zflat),len(zflat)))
        for j in range(len(zflat)):
            a=z.copy();b=z.copy()
            ai=a[m.M];bi=b[m.M];ai[j]+=eps;bi[j]-=eps;a[m.M]=ai;b[m.M]=bi
            jac[:,j]=(m.reconstruct(a)[m.M]-m.reconstruct(b)[m.M])/(2*eps)
        _,q=m._dyn()
        self.assertAlmostEqual(np.linalg.slogdet(jac)[1],.5*np.log(q[m.M]).sum(),places=6)

    def test_transformed_density_including_state_and_parameter_jacobians(self):
        m=self.model;z=m.rng.normal(size=m.psi.shape);omega=np.full_like(z,.3)
        values=[]
        for lam,kappa,sigma in ((.01,-1,.1),(.2,.6,.9),(2.,1.3,3.)):
            m.lam,m.kappa,m.sigma2=lam,kappa,sigma
            target,psi=m.innovation_target(z,omega)
            _,q=m._dyn()
            # Centered transition target + PG likelihood + state Jacobian
            # equals the innovation target minus sum(z²)/2 (constant in theta).
            centered=m._centered_target(psi)
            pg=np.sum(((m.R-.5)*psi-.5*omega*psi**2)[m.M])
            values.append(centered+pg+.5*np.log(q[m.M]).sum()-target)
            # Independent transformed IG/lognormal prior calculation.
            a,b=m.sigma2_prior;mu,sd=m.loglam_prior
            physical=-.5*((np.log(lam)-mu)/sd)**2-np.log(lam)-.5*kappa*kappa-(a+1)*np.log(sigma)-b/sigma
            self.assertAlmostEqual(m.dynamic_prior(),physical+np.log(lam)+np.log(sigma))
        np.testing.assert_allclose(values,-.5*np.sum(z[m.M]**2),atol=1e-8)

    def test_control_constants_and_frozen_adaptation(self):
        for fixed in ({'lam':.05,'sigma2':.5,'kappa':0},{'lam':.05,'kappa':0},{'kappa':0},{}):
            m=InnovationModel(self.seqs,seed=81,sampler='joint_blocked',interweave=True,fixed=fixed)
            samples=m.run(120,60)
            for k,v in fixed.items(): np.testing.assert_array_equal(samples[k],v)
            self.assertTrue(all(np.isfinite(v).all() for v in samples.values()))
            self.assertEqual(m.innovation_adaptations,60 if len(fixed)<3 else 0)
            step=m.innovation_step
            for it in range(3): m._innovation_move(it+120,False)
            self.assertEqual(m.innovation_step,step)
            z=m.innovations(m.psi)
            np.testing.assert_allclose(m.reconstruct(z),m.psi,atol=1e-10)

    def test_rejected_move_restores_joint_state_and_output_guards(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .investigate import execute
        from .investigation_report import summarize
        m=self.model
        m._omega=np.full_like(m.psi,.3)
        old=m.psi.copy();theta=(m.lam,m.kappa,m.sigma2)
        class RejectRng:
            def standard_normal(self): return -100.
            def random(self): return .5
        m.rng=RejectRng()
        m._innovation_move(100,False)
        self.assertEqual(theta,(m.lam,m.kappa,m.sigma2))
        np.testing.assert_array_equal(old,m.psi)
        with TemporaryDirectory() as folder:
            path=Path(folder)
            sentinel=path/'keep.txt';sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):execute(path)
            self.assertEqual(sentinel.read_text(),'preserve')
            (path/'status.json').write_text('{"state":"running","fits":1}')
            with self.assertRaises(ValueError):summarize(path,path/'curated')
            self.assertFalse((path/'curated').exists())

    def test_checkpoint_json_is_verified_without_overwrite(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .investigate import verified_json, read_checkpoint_csv
        with TemporaryDirectory() as folder:
            path=Path(folder)/'dataset.json'
            value={'r':np.array([0,1]),'dt':np.array([1.,.5])}
            verified_json(path,value)
            before=path.read_bytes()
            verified_json(path,value)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):verified_json(path,{'r':[1,1]})
            self.assertEqual(path.read_bytes(),before)
            csv=Path(folder)/'results.csv'
            csv.write_text('fit,passed,cpu\nreal-0-full-centered,True,2.5\n')
            self.assertEqual(read_checkpoint_csv(csv),[{'fit':'real-0-full-centered','passed':True,'cpu':2.5}])

    def test_recovery_export_and_partial_fit_draw_verification(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .investigate import recovery_record, verified_npz
        row=dict(parameter='lam',mean=.3,low=.1,high=.5,passed=False)
        recovered=recovery_record('nested-full',row,.35)
        self.assertEqual(recovered['parameter'],'lam')
        self.assertAlmostEqual(recovered['error'],-.05)
        self.assertTrue(recovered['covered'])
        self.assertFalse(recovered['passed'])
        with TemporaryDirectory() as folder:
            path=Path(folder)/'draws.npz'
            samples={'b0':np.arange(24).reshape(4,6)}
            verified_npz(path,samples);before=path.read_bytes()
            verified_npz(path,samples)
            self.assertEqual(before,path.read_bytes())
            with self.assertRaises(ValueError):verified_npz(path,{'b0':samples['b0']+1})
            self.assertEqual(before,path.read_bytes())

    def test_nested_generator_uses_timing_not_observed_outcomes(self):
        template=[{**s,'item':j} for j,s in enumerate(self.seqs)]
        changed=[{**s,'r':1-s['r'],'x':1-s['x']} for s in template]
        a,truth,states=nested_simulate(template,99)
        b,_,_=nested_simulate(changed,99)
        self.assertEqual(len(truth['beta']),len(template))
        for old,s,t in zip(template,a,b):
            np.testing.assert_array_equal(old['dt'],s['dt'])
            np.testing.assert_array_equal(s['r'],t['r'])
            np.testing.assert_array_equal(s['x'],np.r_[0,s['r'][:-1]])
            self.assertEqual(old['user'],s['user'])
        m=InnovationModel(a,seed=81,sampler='joint_blocked')
        sm=m.run(60,30)
        stacked={k:np.stack([v]*4) for k,v in sm.items()}
        out=nested_summaries(scalar_draws(stacked,'full'),stacked,a)
        self.assertIn('user_level[0]',out)
        for user in (0,1):
            cards=[s['item'] for s in a if s['user']==user]
            np.testing.assert_allclose(sum(out[f'within_user_card[{i}]'] for i in cards),0,atol=1e-12)


if __name__=='__main__':unittest.main()
