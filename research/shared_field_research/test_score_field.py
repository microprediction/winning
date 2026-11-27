import numpy as np
from scipy.special import ndtr
from score_field import prepare,field,sample,moments

def test_score():
    mu=np.array([-.3,.1,.8]); C=np.array([[1,.1,-.05],[.1,1,.02],[-.05,.02,1]])
    a=prepare(C); sd,K,J,c,ev=a; p0,dp=field(mu,a)
    from scipy.integrate import quad
    from scipy.stats import norm
    def price(R):
        C=sd[:,None]*R*sd[None,:]; out=[]
        for i in range(3):
            j,k=[x for x in range(3) if x!=i]; v=C[j,j]+C[i,i]-2*C[j,i]; w=C[k,k]+C[i,i]-2*C[k,i]; r=(C[j,k]-C[j,i]-C[k,i]+C[i,i])/np.sqrt(v*w)
            x=(mu[j]-mu[i])/np.sqrt(v); y=(mu[k]-mu[i])/np.sqrt(w)
            out.append(quad(lambda z:norm.pdf(z)*ndtr((y-r*z)/np.sqrt(1-r*r)),-np.inf,x,epsabs=1e-12)[0])
        return np.array(out)
    eps=1e-4; np.testing.assert_allclose(p0,price(np.eye(3)),atol=1e-11)
    np.testing.assert_allclose(dp,(price(np.eye(3)+eps*K)-price(np.eye(3)-eps*K))/(2*eps),atol=1e-10)
    p,v=sample(mu,a,p0,dp,16,41); np.testing.assert_allclose(p[1],price(np.eye(3)+K),atol=2e-5)
    assert abs(dp.sum())<1e-11

def test_invariance():
    C=np.array([[1.,.1,-.05],[.1,1.,.02],[-.05,.02,1.]]); mu=np.array([-.3,.1,.8]); a=prepare(C); p0,dp=field(mu,a)
    p,_=sample(mu,a,p0,dp,10,1)
    b=prepare(9*C+7); q0,dq=field(3*mu+4,b); q,_=sample(3*mu+4,b,q0,dq,10,1)
    np.testing.assert_allclose(p,q,atol=1e-12)
    perm=[2,0,1]; q0,dq=field(mu[perm],prepare(C[np.ix_(perm,perm)])); np.testing.assert_allclose(q0,p0[perm]);np.testing.assert_allclose(dq,dp[perm])

def test_independent():
    C=np.eye(4); mu=np.arange(4)/3; a=prepare(C); p0,dp=field(mu,a); p,v=sample(mu,a,p0,dp,9,4)
    np.testing.assert_allclose(p,np.stack([p0,p0]),atol=1e-14);assert max(v)<1e-28
