"""Experimental centered/innovation interweaving for the unchanged OU target.

Kept out of the product sampler menu until the diagnostic pilot is evaluated.
Fixed-parameter controls are explicitly different fitted models.
"""
import numpy as np
from .model import DiagnosticModel


class InnovationModel(DiagnosticModel):
    def __init__(self, *args, fixed=None, interweave=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed = dict(fixed or {})
        if set(self.fixed)-{'lam','sigma2','kappa'}:
            raise ValueError('Only OU parameters may be fixed in these controls.')
        for name, value in self.fixed.items():
            if not np.isfinite(value) or (name in ('lam','sigma2') and value <= 0):
                raise ValueError('Invalid fixed OU parameter.')
            if name == 'lam' and not self.lam_bounds[0] < value < self.lam_bounds[1]:
                raise ValueError('Fixed lambda outside support.')
            if name == 'kappa' and abs(value) >= 3:
                raise ValueError('Fixed kappa outside support.')
            setattr(self, name, float(value))
        self.interweave = interweave
        self.innovation_step = 0.15
        self.innovation_adaptations = 0
        self.innovation_acceptance = []

    def _ffbs(self, omega, phi, q, mu):
        self._omega = omega
        return super()._ffbs(omega, phi, q, mu)

    def innovations(self, psi):
        phi, q = self._dyn()
        previous = np.zeros_like(psi)
        previous[:, 1:] = psi[:, :-1]
        z = (psi-phi*previous-(1-phi)*self._mu())/np.sqrt(q)
        return np.where(self.M, z, 0.)

    def reconstruct(self, z):
        phi, q = self._dyn()
        mu = self._mu()
        psi = np.zeros_like(z)
        previous = np.zeros(self.N)
        for t in range(self.Tm):
            current = phi[:,t]*previous+(1-phi[:,t])*mu[:,t]+np.sqrt(q[:,t])*z[:,t]
            psi[:,t] = np.where(self.M[:,t],current,0.)
            previous = psi[:,t]
        return psi

    def dynamic_prior(self):
        # Density in d log(lambda) d kappa d log(sigma2).
        # The lognormal lambda prior already lives in log(lambda).
        # IG(a,b) density times d sigma2/d log(sigma2) gives -a log(sigma2)-b/sigma2.
        if not np.isfinite([self.lam,self.kappa,self.sigma2]).all() or not self.lam_bounds[0] < self.lam < self.lam_bounds[1] or abs(self.kappa)>=3 or self.sigma2<=0:
            return -np.inf
        m,sd = self.loglam_prior
        a,b = self.sigma2_prior
        return (-.5*((np.log(self.lam)-m)/sd)**2
                -.5*(self.kappa/self.kappa_prior_sd)**2-a*np.log(self.sigma2)-b/self.sigma2)

    def innovation_target(self, z, omega):
        prior = self.dynamic_prior()
        if not np.isfinite(prior): return -np.inf, self.psi
        psi = self.reconstruct(z)
        # Gaussian transition determinants cancel the triangular state Jacobian
        # prod sqrt(q_t), including the unchanged first-state variance.
        # Standard-normal z density is constant in this parameter-only move.
        lp = np.sum(((self.R-.5)*psi-.5*omega*psi**2)[self.M])
        return float(lp+prior), psi

    def _centered_target(self, psi):
        phi,q = self._dyn()
        prev=np.zeros_like(psi); prev[:,1:]=psi[:,:-1]
        e=psi-phi*prev-(1-phi)*self._mu()
        return float(-.5*np.sum((np.log(q)+e*e/q)[self.M])+self.dynamic_prior())

    def _update_lam_sigma2(self, psi, it, adapt):
        if not self.fixed:
            super()._update_lam_sigma2(psi,it,adapt)
        elif set(self.fixed)=={'lam','kappa'}:
            # Exact conjugate diffusion conditional for a fixed decay/practice control.
            phi,q=self._dyn()
            prev=np.zeros_like(psi); prev[:,1:]=psi[:,:-1]
            e=(psi-phi*prev-(1-phi)*self._mu())[self.TR]
            h=q[self.TR]/self.sigma2
            a,b=self.sigma2_prior
            self.sigma2=float(1/self.rng.gamma(a+self.TR.sum()/2,1/(b+.5*np.sum(e*e/h))))
            self.acc_rate=np.nan
        elif set(self.fixed)=={'kappa'}:
            # Original collapsed lambda + exact sigma update, kappa held constant.
            use=self.use_kappa; self.use_kappa=False
            try: super()._update_lam_sigma2(psi,it,adapt)
            finally: self.use_kappa=use
        elif set(self.fixed)=={'lam','sigma2','kappa'}:
            self.acc_rate=np.nan
        else:
            raise ValueError('Unsupported fixed-parameter control.')
        if self.interweave:
            self._innovation_move(it,adapt)

    def _innovation_move(self,it,adapt):
        z=self.innovations(self.psi)
        current,_=self.innovation_target(z,self._omega)
        names=[k for k in ('lam','kappa','sigma2') if k not in self.fixed]
        if not names: return
        accepted=0
        for _ in range(self.n_mh):
            old={k:getattr(self,k) for k in names}
            for k in names:
                delta=self.innovation_step*self.rng.standard_normal()
                setattr(self,k,old[k]+delta if k=='kappa' else float(np.exp(np.log(old[k])+delta)))
            proposed,candidate=self.innovation_target(z,self._omega)
            if np.log(self.rng.random()) < proposed-current:
                current=proposed; self.psi=candidate; accepted+=1
            else:
                for k,v in old.items(): setattr(self,k,v)
        rate=accepted/self.n_mh
        if adapt:
            self.innovation_adaptations+=1
            self.innovation_step=float(np.clip(self.innovation_step*np.exp((rate-.234)/(it+1)**.6),.001,2.))
        else: self.innovation_acceptance.append(rate)

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.sampling_info.update(interweave=self.interweave, fixed=self.fixed,
                                 innovation_step=self.innovation_step,
                                 innovation_adaptations=self.innovation_adaptations,
                                 innovation_acceptance=float(np.mean(self.innovation_acceptance)) if self.innovation_acceptance else None)
        return result
