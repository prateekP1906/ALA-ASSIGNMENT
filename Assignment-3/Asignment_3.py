# ----------------------------------------------------------------------------
# AI assistance disclosure
# Tool     : Claude Sonnet 5.5 (Anthropic), claude.ai
# Date     : 5 November 2026
# Used for : Initial NumPy implementation of Algorithms 4, 5 and 6 and the
#            random-parameter test harness (init_mha_params).
# Author's contribution: Selection of GPT-2 small hyperparameters, the choice
#            of algorithms, review of the code against Phuong & Hutter (2022),
#            and verification of the tests (e.g. parameter count 2,362,368).
# Reference: Phuong & Hutter, "Formal Algorithms for Transformers",
#            arXiv:2207.09238 (2022).
# ----------------------------------------------------------------------------


import numpy as np


L_MAX = 1024     # maximum sequence length
D_E = 768        # model width
N_LAYERS = 12    # L
N_HEADS = 12     # H
D_MLP = 3072
N_V = 50257      # vocabulary size
D_ATTN = D_E // N_HEADS   # 64 = d_attn = d_mid (per head)
D_MID = D_ATTN
D_OUT = D_E               # output width of MHAttention (Alg. 5)


def softmax(A, axis = 0):
    """softmax(A)[tz, tx] = exp A[tz,tx] / sum_t exp A[t,tx]   (paper eq. 2)."""
    A = A - np.max(A, axis = axis, keepdims = True)
    E = np.exp(A)
    return E/np.sum(E, axis=axis, keepdims=True)

def causal_mask(l_z, l_x):
    """Mask[tz, tx] = [[tz <= tx]]  (paper eq. 3, unidirectional attention)."""

    tz = np.arange(l_z)[:, None]
    tx = np.arange(l_x)[None, :]
    return (tz <= tx).astype(np.int8)

#### ALGORITHM 1  : TOKEN EMBEDDING ###

def token_embedding(v, We):
    """
    Input : v  token ID (int, 0-indexed), or an integer array of IDs
            We in R^{d_e x N_V}, token embedding matrix
    Output: e  in R^{d_e}  (or R^{d_e x len(v)} if v is an array)
    """

    return We[:, v]

#### ALGORITHM 6 : LAYER_NORM ####

def layer_norm(e, gamma, beta, eps = 0.0):

     """
    Input : e in R^{d_e} (or R^{d_e x l}: each column normalised independently)
            gamma, beta in R^{d_e}
    Output: e_hat in same shape as e
    eps = 0 reproduces the paper exactly; GPT-2's released code uses eps = 1e-5.
    """
     
     e = np.asarray(e, dtype = np.float64)
     d_e = e.shape[0]
     m = np.sum(e, axis = 0, keepdims = True) / d_e
     v = np.sum((e - m) ** 2, axis = 0, keepdims = True) / d_e
     g = gamma.reshape((-1,) + (1,) * (e.ndim - 1))
     b = beta.reshape((-1,) + (1,) * (e.ndim - 1))
     return ( e - m ) / np.sqrt(v + eps) * g + b

#### ALGORITHM 4 : ATTENTION (SINGLE HEAD) ####

def attention(X,Z,Wq,bq,Wk,bk,Wv,bv, mask = None):
    """
    Input : X in R^{d_x x l_x}, Z in R^{d_z x l_z}
    Params: Wq in R^{d_attn x d_x}, bq in R^{d_attn}
            Wk in R^{d_attn x d_z}, bk in R^{d_attn}
            Wv in R^{d_out  x d_z}, bv in R^{d_out}
    mask  : {0,1}^{l_z x l_x}; None means Mask == 1 (bidirectional / cross-attn)
    Output: V_tilde in R^{d_out x l_x}
    """
    d_attn = Wq.shape[0]
    l_x, l_z = X.shape[1], Z.shape[1]

    Q = Wq @ X + bq[:, None]
    K = Wk @ Z + bk[:, None]
    V = Wv @ Z + bv[:, None]
    S = K.T @ Q

    if mask is not None : 
        assert mask.shape == (l_z, l_x)
        S = np.where(mask.astype(bool), S, -np.inf)
    
    return V @ softmax(S / np.sqrt(d_attn), axis = 0)


####ALGORITHM 5 : MHAttention ######

def mh_attention(X, Z, W, mask = None):
    """
    Input : X in R^{d_x x l_x}, Z in R^{d_z x l_z}
    Params: W = {"heads": [ {Wq,bq,Wk,bk,Wv,bv} for h in range(H) ],
                 "Wo": R^{d_out x H*d_mid}, "bo": R^{d_out}}
    Output: V_tilde in R^{d_out x l_x}
    """

    Y = [attention(X, Z, h["Wq"], h["bq"], h["Wk"], h["bk"], h["Wv"], h["bv"], mask)
         for h in W["heads"]]
    Y = np.concatenate(Y, axis = 0)
    return W["Wo"] @ Y + W["bo"][:, None]

#### Random GPT - 2 - small - shaped parameters (for testing)

def init_mha_params(rng, H=N_HEADS, d_x=D_E, d_z=D_E, d_attn=D_ATTN, d_mid=D_MID,
                    d_out=D_OUT, scale=0.02):
    heads = [dict(Wq=rng.normal(0, scale, (d_attn, d_x)), bq=rng.normal(0, scale, d_attn),
                  Wk=rng.normal(0, scale, (d_attn, d_z)), bk=rng.normal(0, scale, d_attn),
                  Wv=rng.normal(0, scale, (d_mid, d_z)),  bv=rng.normal(0, scale, d_mid))
             for _ in range(H)]
    return dict(heads=heads,
                Wo=rng.normal(0, scale, (d_out, H * d_mid)),
                bo=rng.normal(0, scale, d_out))

### TESTS #####

# Self-tests: basic test cases written by the author; refined and extended
# with AI assistance (Claude Sonnet 5.5, Anthropic).

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # HYPERPARAMETER CONSISTENCY
    assert N_HEADS * D_MID == D_E and D_ATTN == 64

    # Algorithm 1
    We = rng.normal(0,00.2, (D_E, N_V))
    assert token_embedding(464, We).shape == (D_E,)
    assert np.allclose(token_embedding(464,We), We[:,464])
    assert token_embedding(np.array([1,2,3]), We).shape == (D_E, 3)

    #Algorithm 6
    e = rng.normal(3.0, 5.0, D_E)
    out = layer_norm(e, np.ones(D_E), np.zeros(D_E))
    assert abs(out.mean()) < 1e-10 and abs(out.var() - 1) < 1e-10
    g, b = rng.normal(size=D_E), rng.normal(size=D_E)
    Em = rng.normal(size=(D_E, 5))
    cols = np.stack([layer_norm(Em[:, i], g, b) for i in range(5)], axis=1)
    assert np.allclose(layer_norm(Em, g, b), cols)

    #Algorithm 4
    l = 7
    X = rng.normal(size=(D_E, l))
    W = init_mha_params(rng)
    M = causal_mask(l, l)
    assert M[0, 0] == 1 and M[1, 0] == 0 and M[0, 1] == 1

    #Algorithm 5

    h0 = W["heads"][0]
    Y0 = attention(X, X, **h0, mask = M)
    assert Y0.shape == (D_MID, l)

    out = mh_attention(X, X, W, M)
    assert out.shape == (D_OUT, l)

    #Causality
    X2 = X.copy(); X2[:, -1] += 10.0
    out2 = mh_attention(X2, X2, W, M)
    assert np.allclose(out[:, :-1], out2[:, :-1]) and not np.allclose(out[:, -1], out2[:, -1])
 

    print("TEST PASSED!!")