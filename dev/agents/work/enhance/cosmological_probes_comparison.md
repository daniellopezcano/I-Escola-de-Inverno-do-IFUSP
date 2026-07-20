# Observational Probes of Cosmology: A Comparative Guide

*Lecture material — comparative anatomy of seven cosmological probes*

---

## 0. How to read this document

Every cosmological probe can be described by answering the same five questions:

1. **What is observed?** — the raw field, and what it traces.
2. **Where and when?** — redshift, scales, regime (linear/nonlinear).
3. **What does it constrain?** — parameters, and crucially the *degeneracy direction*.
4. **How is it measured?** — instrument, statistic, radial resolution.
5. **How is it modelled?** — forward model, nuisance parameters, inference scheme, limiting systematic.

The organising principle behind all of it is a single distinction that is worth
putting on the board before anything else:

> A cosmological probe measures either the **geometry** of the Universe
> (distances, expansion rate), the **growth** of structure within it
> (the amplitude and rate of clustering), or some combination of the two.
>
> Within $\Lambda$CDM + GR, geometry predicts growth. Measuring both
> independently is therefore not merely a precision exercise — it is a
> *test of the framework itself*.

A second organising distinction, almost as useful:

> Some probes see **total matter** (lensing, CMB lensing, gravitational-wave
> propagation). Others see a **biased tracer** of it (galaxies, neutral
> hydrogen, the intergalactic medium). The former are cleaner but noisier;
> the latter are precise but require a bias model.

The seven probes covered here are: galaxy clustering, weak gravitational
lensing, the primary CMB, CMB lensing, the Lyman-$\alpha$ forest,
gravitational waves (standard sirens), and 21 cm intensity mapping.

---

## 1. Galaxy Clustering

### 1.1 What is observed

The three-dimensional number density field of galaxies,
$\delta_g(\mathbf{x}) = n_g(\mathbf{x})/\bar{n}_g - 1$, reconstructed from a
spectroscopic redshift catalogue. Galaxies are a **biased, discrete tracer**
of the underlying total matter field: on large scales
$\delta_g \simeq b_1 \delta_m$, with higher-order bias terms entering as
nonlinearity grows.

Because redshifts are used as distance proxies, peculiar velocities distort
the inferred positions. This is not a nuisance but a *signal*: redshift-space
distortions (RSD) encode the velocity field, hence the growth rate of
structure.

### 1.2 Where and when

| | |
|---|---|
| Redshift range | $z \sim 0 - 2.1$ (DESI LRG/ELG/QSO); Euclid H$\alpha$ emitters $z \sim 0.9 - 1.8$ |
| Effective redshift | Analysis is done in tomographic bins with $z_{\rm eff} \sim 0.3, 0.5, 0.7, 0.9, 1.3, 2.1$ |
| Scales in $k$ | $k \sim 0.005 - 0.25\ h\,{\rm Mpc}^{-1}$ (full-shape EFT analyses); BAO feature at $r_d \simeq 147$ Mpc |
| Regime | Linear to *mildly* nonlinear; the $k_{\rm max}$ cut is chosen precisely to stay where perturbation theory is trustworthy |
| Volume | DESI: $\sim 14{,}000$ deg$^2$, tens of millions of redshifts, effective volume $\sim 100\ (h^{-1}{\rm Gpc})^3$ |

### 1.3 What it constrains

- **Geometry**: the BAO feature acts as a standard ruler, delivering
  $D_M(z)/r_d$ (transverse) and $D_H(z)/r_d = c/[H(z) r_d]$ (radial), and
  their combination $D_V/r_d$.
- **Growth**: RSD deliver the combination $f\sigma_8(z)$, where
  $f = d\ln D/d\ln a \simeq \Omega_m(z)^{0.55}$ in GR.
- **Shape**: the broadband power spectrum constrains $\Omega_m h^2$ through
  the turnover/equality scale, and $n_s$.
- **Extensions**: $\sum m_\nu$ (free-streaming suppression), $w_0$–$w_a$,
  and primordial non-Gaussianity $f_{\rm NL}^{\rm local}$ via the
  scale-dependent bias $\Delta b(k) \propto f_{\rm NL}/k^2$ on the very
  largest scales.

**Degeneracy direction:** BAO alone constrains $\Omega_m$ and $H_0 r_d$
tightly but says nothing about the amplitude; the amplitude enters through
$f\sigma_8$ and through the bias-degenerate combination $b_1\sigma_8$.

### 1.4 How it is measured

| | |
|---|---|
| Apparatus | Massively multiplexed fibre-fed spectrographs (DESI: 5000 robotic fibres; Euclid: slitless NISP grism) |
| Experiments | SDSS/BOSS/eBOSS, DESI, Euclid, 4MOST, PFS |
| Statistic fitted | Power-spectrum multipoles $P_{0,2,4}(k)$ or correlation-function multipoles $\xi_{0,2,4}(s)$; increasingly also the bispectrum |
| Radial resolution | Excellent — spectroscopic, $\sigma_z/(1+z) \sim 10^{-4}$, enabling genuine 3D analysis |
| Noise floor | Shot noise $1/\bar n$ on small scales; cosmic variance on large scales |

### 1.5 How it is modelled

- **Forward model**: Boltzmann code (CAMB/CLASS) for the linear
  $P_{\rm lin}(k)$, then one-loop perturbation theory — EFT of LSS,
  or TNS/Lagrangian schemes — for the redshift-space galaxy spectrum.
- **Nuisance parameters**: per redshift bin, typically $b_1, b_2, b_{s^2},
  b_{3\rm nl}$, EFT counterterms $c_0, c_2, c_4$, and stochastic terms
  $P_{\rm shot}, a_0, a_2$. Roughly **7–10 nuisance parameters per tracer
  per bin** against $\sim 5$ cosmological parameters — one of the least
  favourable nuisance ratios of any probe.
- **Inference**: Gaussian likelihood with covariance estimated from
  thousands of approximate mocks (EZmocks, PATCHY, GLAM), or from analytic
  Gaussian+trispectrum covariances. Hartlap/Percival corrections apply.
- **Dominant systematics**: imaging/target-selection systematics on large
  scales, fibre collisions and redshift failures on small scales, and the
  theoretical error of the perturbative model at $k_{\rm max}$.
- **Limiting factor**: theoretical modelling — pushing $k_{\rm max}$ up is
  where all the remaining information is, and where the model breaks.

---

## 2. Weak Gravitational Lensing (Cosmic Shear)

### 2.1 What is observed

Coherent distortions in the observed shapes of background galaxies, induced
by the gravitational potential of foreground structure. The measured
quantity is the ellipticity of each galaxy, an extremely noisy estimator of
the shear $\gamma$ (intrinsic shape noise $\sigma_e \sim 0.26$ per component,
against a signal $\gamma \sim 10^{-2}$). Signal emerges only statistically,
by averaging over $10^8$ galaxies.

Crucially, lensing responds to the **total matter field**, dark and baryonic
alike, with no bias parameter interposed. It is the closest thing we have to
a direct image of the projected mass distribution.

### 2.2 Where and when

| | |
|---|---|
| Source redshifts | $z_s \sim 0.2 - 1.5$ (DES, KiDS, HSC); LSST/Euclid extend to $z_s \sim 2 - 3$ |
| Effective lens redshift | The lensing efficiency kernel $W(\chi)$ peaks roughly halfway to the sources, so structure at $z \sim 0.3 - 0.6$ dominates |
| Scales | $\ell \sim 100 - 3000$, i.e. $\theta \sim 0.5' - 250'$; corresponds to $k \sim 0.1 - 5\ h\,{\rm Mpc}^{-1}$ |
| Regime | Deeply **nonlinear**. This is the defining difficulty of the probe |
| Sky | DES Y3: 4143 deg$^2$; KiDS-1000: 1006 deg$^2$; LSST: 18,000 deg$^2$ planned |

Note the projection: because information from a broad range of $\chi$ is
compressed onto one angular scale (the Limber approximation,
$k \simeq (\ell + 1/2)/\chi$), a single $\ell$ mixes many physical scales.
This is worth deriving explicitly in a lecture — it is the reason lensing is
a *tomographic* rather than a 3D probe.

### 2.3 What it constrains

- **Growth, almost exclusively.** The amplitude of the shear correlation
  scales roughly as $\sigma_8^2 \Omega_m^{1.5}$, so the well-measured
  combination is
  $$S_8 \equiv \sigma_8 \sqrt{\Omega_m/0.3}$$
  with a characteristic "banana" degeneracy in the $\Omega_m$–$\sigma_8$
  plane.
- **Extensions**: $w_0$–$w_a$ (through the growth history and the geometric
  kernel), $\sum m_\nu$, and modified gravity via the lensing potential
  parameter $\Sigma$.
- Weak sensitivity to $H_0$, $\Omega_b$, $n_s$ — these are typically kept in
  wide priors or fixed to CMB values.

**Why it matters pedagogically:** the "$S_8$ tension", in which lensing
surveys have consistently preferred lower amplitudes ($S_8 \approx 0.76-0.79$)
than the CMB extrapolation ($S_8 \approx 0.83$), is the textbook example of
a geometry-vs-growth comparison doing real work. Recent results (KiDS-Legacy,
DES+KiDS joint analyses) have narrowed the discrepancy considerably, which is
itself an instructive lesson about systematics.

### 2.4 How it is measured

| | |
|---|---|
| Apparatus | Wide-field imaging cameras with excellent, stable PSF — DECam, OmegaCAM, HSC, LSSTCam, Euclid VIS |
| Experiments | DES, KiDS, HSC-SSP; LSST, Euclid, Roman |
| Statistic fitted | Real-space two-point functions $\xi_\pm(\theta)$ between tomographic bins, or $C_\ell^{\kappa\kappa}$; increasingly "3×2pt" (shear–shear, galaxy–shear, galaxy–galaxy) |
| Radial resolution | Poor — **photometric** redshifts, $\sigma_z \sim 0.05(1+z)$ at best. Only 4–6 broad tomographic bins are meaningful |
| Noise floor | Shape noise, hence the relentless push for source density $n_{\rm eff}$ (arcmin$^{-2}$) |

### 2.5 How it is modelled

- **Forward model**: linear $P(k)$ from a Boltzmann code $\to$ nonlinear
  $P(k)$ from HALOFIT, HMcode, or an emulator (EuclidEmulator2, BACCO,
  CosmicEmu) $\to$ baryonic feedback correction $\to$ Limber projection with
  the tomographic lensing kernels.
- **Nuisance parameters**: multiplicative shear bias $m_i$ per bin;
  photometric redshift shifts $\Delta z_i$ (and sometimes width/shape
  parameters); intrinsic alignments (NLA: $A_{\rm IA}, \eta_{\rm IA}$; TATT
  adds $A_2, b_{\rm TA}$); baryonic feedback amplitude ($A_{\rm bary}$,
  $T_{\rm AGN}$, or halo-model $c_{\rm min}$). Typically **10–20 nuisance
  parameters** for $\sim 5$ cosmological ones.
- **Inference**: Gaussian likelihood, analytic covariance (halo model +
  survey geometry), nested sampling (MultiNest, PolyChord, Nautilus).
  Simulation-based inference is now being applied to non-Gaussian statistics
  (peak counts, wavelet scattering transforms).
- **Dominant systematics**: intrinsic alignments and baryonic feedback are
  the two that limit $k_{\rm max}$; photo-$z$ calibration is the one that has
  historically shifted central values the most.
- **Limiting factor**: astrophysical modelling of the nonlinear regime, not
  statistics.

---

## 3. The Primary CMB

### 3.1 What is observed

Temperature and linear polarisation anisotropies of the microwave sky,
$\Delta T/T \sim 10^{-5}$, imprinted at the surface of last scattering. The
observable is a **two-dimensional field on a sphere at essentially a single
redshift**, decomposed into $T$, $E$ and (so far only via lensing) $B$ modes.

Physically, the anisotropies are acoustic oscillations in the
photon–baryon plasma, driven by primordial perturbations and terminated at
recombination. The physics is linear, well understood, and computable to
sub-percent accuracy from first principles.

### 3.2 Where and when

| | |
|---|---|
| Redshift | $z_* \simeq 1090$ (last scattering), with the reionisation bump at $z \sim 6-10$ in large-scale polarisation |
| Scales | $\ell \sim 2 - 2500$ (Planck); ACT DR6 and SPT-3G extend to $\ell \sim 4000-8000$ in TT/TE/EE |
| Physical scales | Sound horizon $r_s(z_*) \simeq 145$ Mpc; damping scale $\sim 10$ Mpc |
| Regime | **Linear.** This is the probe's superpower |
| Sky | Full sky (Planck, $f_{\rm sky} \sim 0.7$ after masking); ground-based experiments trade area for depth and resolution |

### 3.3 What it constrains

The primary CMB is the anchor of modern cosmology because it constrains six
$\Lambda$CDM parameters simultaneously and with very different physics:

| Parameter | Physical origin |
|---|---|
| $\theta_*$ | Angular size of the sound horizon — pure geometry, measured to $\sim 0.03\%$ |
| $\Omega_b h^2$ | Odd/even acoustic peak height ratio |
| $\Omega_c h^2$ | Overall peak envelope, early ISW, radiation driving |
| $A_s$ | Overall amplitude |
| $n_s$ | Tilt across the peak sequence |
| $\tau$ | Large-scale EE reionisation bump; degenerate with $A_s$ as $A_s e^{-2\tau}$ |

$H_0$ and $\Omega_m$ are **derived**, not directly measured: they follow from
$\theta_*$ plus the matter densities *under the assumption of a cosmological
model for the late-time expansion*. This is why the CMB's $H_0$ value is
model-dependent in a way that a local measurement is not, and it is the
crux of the Hubble tension.

- **Extensions**: $N_{\rm eff}$, $\sum m_\nu$ (weakly, mostly through
  lensing), spatial curvature $\Omega_k$ (degenerate with $H_0$ from primary
  CMB alone), tensor-to-scalar ratio $r$ from primordial $B$ modes,
  isocurvature modes, running $dn_s/d\ln k$.

**Calibration status:** the CMB is one of only two **absolutely calibrated**
probes on this list (the other being gravitational waves). The sound horizon
$r_d$ is a physically computed length, not an empirically calibrated one.

### 3.4 How it is measured

| | |
|---|---|
| Apparatus | Cryogenic bolometer arrays (TES detectors) at $\sim 100$ mK, on satellites or high-altitude dry sites (Atacama, South Pole) |
| Experiments | COBE, WMAP, Planck; ACT, SPT-3G, POLARBEAR/Simons Array; Simons Observatory, CMB-S4, LiteBIRD |
| Statistic fitted | Angular power spectra $C_\ell^{TT}, C_\ell^{TE}, C_\ell^{EE}$ (+ $C_\ell^{BB}$) |
| Radial resolution | None — it is a single screen. All 3D information comes from lensing and secondary anisotropies |
| Noise floor | Detector noise at high $\ell$; **cosmic variance** at low $\ell$ (an irreducible limit: only $2\ell+1$ modes per multipole) |

### 3.5 How it is modelled

- **Forward model**: a linear Boltzmann solver (CAMB, CLASS) integrating the
  coupled Einstein–Boltzmann hierarchy through recombination (RECFAST,
  CosmoRec, HyRec). No perturbation theory beyond linear order is needed.
- **Nuisance parameters**: astrophysical foregrounds (thermal and kinetic SZ,
  cosmic infrared background, radio and dusty point sources, Galactic dust
  and synchrotron) and instrumental terms (calibration, beam eigenmodes,
  polarisation efficiency). Planck's `plik` likelihood carries $\sim 20$ such
  parameters; the physics model itself needs none.
- **Inference**: Gaussian or Hamimeche–Lewis likelihood in $C_\ell$;
  MCMC (CosmoMC, Cobaya, MontePython).
- **Dominant systematics**: foreground separation (decisive for $B$ modes),
  bandpass and beam calibration, and the low-$\ell$ optical-depth measurement.
- **Limiting factor**: cosmic variance on large scales — no future experiment
  can improve on Planck at $\ell < 30$ in temperature.

---

## 4. CMB Lensing

### 4.1 What is observed

The **secondary** distortion of the CMB by all the matter between us and
$z_* = 1090$. Lensing remaps the primary anisotropies, introducing a
characteristic non-Gaussianity: correlations between different multipoles
that would be independent in an unlensed sky. From these, a quadratic (or
maximum-likelihood) estimator reconstructs the lensing convergence
$\kappa(\hat{n})$, or equivalently the lensing potential $\phi$.

Conceptually it is the same physics as cosmic shear, with two decisive
advantages: the source plane is a **single, precisely known redshift**, and
the source has **no intrinsic alignments and no shape noise** in the galaxy
sense.

### 4.2 Where and when

| | |
|---|---|
| Source redshift | Exactly $z_* = 1090$ |
| Lensing kernel | Extremely broad, peaking at $z \sim 2$, with substantial contributions from $z \sim 0.5 - 5$ |
| Scales | $L \sim 40 - 1300$ (ACT DR6 baseline); corresponds to $k \sim 0.05 - 0.5\ h\,{\rm Mpc}^{-1}$ |
| Regime | Predominantly **linear to quasi-linear** — this is the key contrast with galaxy weak lensing |
| Sky | Planck: full sky; ACT DR6: $\sim 9400$ deg$^2$; SO/CMB-S4: comparable to larger |

### 4.3 What it constrains

- The lensing power spectrum $C_L^{\phi\phi}$ measures a projected amplitude
  of matter clustering. The best-constrained combination differs from the
  galaxy-lensing one because of the different kernel:
  $$S_8^{\rm CMBL} \equiv \sigma_8 (\Omega_m/0.3)^{0.25}$$
  a *shallower* $\Omega_m$ exponent, so the degeneracy direction is rotated
  relative to cosmic shear — which is exactly why combining them is powerful.
- **$\sum m_\nu$**: currently the single most promising route, since
  neutrino free-streaming suppresses $C_L^{\phi\phi}$ by
  $\sim 1\%$ per 20 meV in a redshift range where the effect is not masked
  by baryons.
- **Delensing** for primordial $B$-mode searches.
- Cross-correlations with galaxy surveys, CIB, and cluster catalogues break
  bias–amplitude degeneracies and localise the growth measurement in
  redshift.

**Result worth quoting in a lecture:** ACT DR6 reports
$\sigma_8(\Omega_m/0.3)^{0.25} = 0.830 \pm 0.014$, in good agreement with the
primary-CMB $\Lambda$CDM extrapolation. That a high-redshift, quasi-linear
lensing measurement agrees with the CMB while low-redshift galaxy lensing
sat somewhat low is a genuinely interesting tension diagnostic.

### 4.4 How it is measured

| | |
|---|---|
| Apparatus | Same instruments as the primary CMB — this is a reanalysis of the same maps, not a separate experiment |
| Experiments | Planck, ACT, SPT-3G; Simons Observatory, CMB-S4 |
| Statistic fitted | $C_L^{\phi\phi}$ from quadratic estimators (Hu–Okamoto) or maximum-likelihood/iterative estimators for deep polarisation data |
| Radial resolution | None — a single broad projection. Redshift information requires cross-correlation |
| Noise floor | Reconstruction noise $N_L^{(0)}$, set by the primary CMB's own sample variance and detector noise |

### 4.5 How it is modelled

- **Forward model**: Boltzmann code $\to$ nonlinear $P(k)$ (matters at the
  $\sim 10\%$ level at high $L$) $\to$ Limber projection with the CMB lensing
  kernel.
- **Nuisance parameters**: remarkably few. The main concerns are biases
  rather than free parameters — $N^{(0)}$, $N^{(1)}$ and higher-order
  reconstruction biases, plus contamination of the estimator by extragalactic
  foregrounds (tSZ, CIB), handled by profile-hardening or shear estimators
  rather than by marginalisation. **The cleanest nuisance budget of any
  late-time probe.**
- **Inference**: Gaussian likelihood in bandpowers, with simulation-based
  bias subtraction; typically blinded analyses.
- **Dominant systematics**: extragalactic foreground bias in the reconstruction.
- **Limiting factor**: reconstruction noise, i.e. raw detector sensitivity —
  which is why SO and CMB-S4 will improve this dramatically.

---

## 5. The Lyman-$\alpha$ Forest

### 5.1 What is observed

The dense series of absorption features blueward of Ly$\alpha$ emission in
the spectra of high-redshift quasars, produced by neutral hydrogen in the
diffuse intergalactic medium. Each quasar sightline is effectively a
**one-dimensional core sample** through the cosmic web.

The tracer here is not galaxies but the *low-density IGM* itself, which
sits close to the mean density and therefore remains comparatively close to
the linear regime in its dynamics, even though the mapping from density to
observed flux is strongly nonlinear:
$$F = e^{-\tau}, \qquad \tau \propto \rho^{2 - 0.7(\gamma - 1)} T^{-0.7}$$
via the fluctuating Gunn–Peterson approximation.

### 5.2 Where and when

| | |
|---|---|
| Redshift | $z \sim 2 - 4.5$ (set by where Ly$\alpha$ falls in the optical window and by IGM opacity) |
| Scales | The **smallest scales of any large-scale-structure probe**: $k_\parallel \sim 0.001 - 0.1\ {\rm s\,km^{-1}}$, i.e. up to $k \sim 5 - 10\ h\,{\rm Mpc}^{-1}$ |
| Regime | Mildly nonlinear in density, but *thermally broadened* — the small-scale cutoff is astrophysical, not gravitational |
| Sky | eBOSS: $\sim 200{,}000$ quasar spectra; DESI will exceed $10^6$ |

### 5.3 What it constrains

- **$P_{\rm 1D}(k_\parallel)$**: the amplitude and slope of the matter power
  spectrum at small scales and high redshift — the longest lever arm in $k$
  available anywhere in cosmology when combined with the CMB.
- **$\sum m_\nu$**: neutrino free-streaming suppression is maximal here, and
  Ly$\alpha$ + CMB gives some of the tightest limits.
- **Warm dark matter / dark matter free-streaming**: the small-scale cutoff
  places the strongest astrophysical bounds on thermal relic WDM masses.
- **Running of the spectral index $dn_s/d\ln k$**, and inflationary features.
- **$P_{\rm 3D}$ and BAO at $z \simeq 2.3$**: from cross-sightline
  correlations and Ly$\alpha$ $\times$ quasar cross-correlations — the
  highest-redshift BAO measurement available.

### 5.4 How it is measured

| | |
|---|---|
| Apparatus | Moderate-resolution multi-object spectrographs for $P_{\rm 3D}$/BAO (SDSS, DESI); high-resolution echelle spectrographs (UVES/VLT, HIRES/Keck, ESPRESSO) for small-scale $P_{\rm 1D}$ |
| Experiments | BOSS/eBOSS, DESI; WEAVE-QSO; high-resolution samples (KODIAQ, SQUAD) |
| Statistic fitted | 1D flux power spectrum $P_{\rm 1D}(k_\parallel, z)$; 3D flux correlation $\xi(r_\parallel, r_\perp)$ |
| Radial resolution | Superb along the line of sight; sparse and shot-noise-limited transverse to it |
| Noise floor | Spectral S/N per pixel and sightline density |

### 5.5 How it is modelled

- **Forward model**: there is **no perturbative shortcut**. The flux field
  must be predicted from cosmological **hydrodynamic simulations**
  (Nyx, MP-Gadget, Sherwood), post-processed and interpolated by an
  emulator over both cosmological and IGM-thermal parameters.
- **Nuisance parameters**: IGM thermal history $T_0(z)$ and $\gamma(z)$
  (temperature–density relation), mean transmitted flux $\bar{F}(z)$, UV
  background fluctuations, patchy reionisation and its residual heating,
  damped Ly$\alpha$ systems, metal-line contamination (SiII, SiIII),
  continuum-fitting errors, spectrograph resolution and noise-power
  subtraction. Frequently **more nuisance than cosmological parameters**.
- **Inference**: Gaussian likelihood on emulated $P_{\rm 1D}$ bandpowers;
  increasingly simulation-based inference given the simulator-driven model.
- **Dominant systematics**: the IGM thermal state (degenerate with the
  small-scale power suppression one is trying to measure) and continuum
  fitting.
- **Limiting factor**: hydrodynamic simulation accuracy and the cost of
  covering the parameter space.

---

## 6. Gravitational Waves (Standard Sirens)

### 6.1 What is observed

The strain $h(t)$ induced by a passing gravitational wave from a coalescing
compact binary, measured as a differential arm-length change in a
kilometre-scale laser interferometer. The observable is a **time series**,
not a map or a catalogue of positions — methodologically this probe stands
apart from everything else in this document.

The cosmological content follows from a remarkable property of the
waveform: general relativity predicts the amplitude of the signal, so the
**luminosity distance $d_L$ is measured directly from the waveform**,
without any calibration ladder. Hence "standard siren".

### 6.2 Where and when

| | |
|---|---|
| Redshift | LVK: $z \lesssim 0.1$ for BNS, $z \lesssim 1$ for BBH. Einstein Telescope / Cosmic Explorer: BBH out to $z \sim 10-100$. LISA: massive black hole binaries at $z \sim 1-10$ |
| Scales | **Not applicable in the usual sense.** GWs do not measure a power spectrum of structure; they measure the background metric and its propagation |
| Regime | Fully relativistic and strong-field at the source; linear perturbation on the background during propagation |
| Sky localisation | Poor — from tens to thousands of deg$^2$, depending on network size |

### 6.3 What it constrains

- **$H_0$, absolutely and directly.** Combining $d_L$ from the waveform with
  a redshift gives the Hubble constant with no distance ladder and no sound
  horizon. This makes GWs a genuinely independent arbiter of the Hubble
  tension.
  - **Bright sirens**: an electromagnetic counterpart gives the host galaxy
    and hence $z$. GW170817 + NGC 4993 gave $H_0 \approx 70^{+12}_{-8}$
    km s$^{-1}$ Mpc$^{-1}$.
  - **Dark sirens**: no counterpart; the redshift is inferred statistically
    from a galaxy catalogue in the localisation volume, or from features in
    the compact-object mass distribution (the "spectral siren" method, using
    the pair-instability mass gap as a mass scale).
- **$\Omega_m$, $w_0$–$w_a$** at higher redshift with third-generation
  detectors, via the $d_L(z)$ relation.
- **Modified gravity in a form no other probe can access**: if gravity
  propagates differently from light, the ratio
  $d_L^{\rm GW}/d_L^{\rm EM} \neq 1$, testing running of the effective Planck
  mass, extra spatial dimensions, and graviton damping. Also the GW
  propagation speed ($|c_{\rm GW}/c - 1| \lesssim 10^{-15}$ from GW170817,
  which eliminated large classes of scalar–tensor theories overnight).

**Calibration status:** absolute. This and the CMB sound horizon are the two
independent absolute rulers in cosmology.

### 6.4 How it is measured

| | |
|---|---|
| Apparatus | Power-recycled Fabry–Pérot Michelson interferometers with squeezed-light injection; strain sensitivity $\sim 10^{-23}\,{\rm Hz}^{-1/2}$ |
| Experiments | LIGO, Virgo, KAGRA; LIGO-India; Einstein Telescope, Cosmic Explorer; LISA (space, mHz band); pulsar timing arrays (nHz band) |
| Statistic fitted | Not a summary statistic — a **population-level hierarchical posterior** over individual event parameters $(m_1, m_2, \chi, d_L, \dots)$ |
| Redshift information | **Absent from the GW data itself.** Mass and redshift are degenerate: only the redshifted chirp mass $\mathcal{M}(1+z)$ is measured |
| Noise floor | Quantum shot noise at high frequency, seismic and thermal noise at low frequency; detector-noise-limited, not cosmic-variance-limited |

### 6.5 How it is modelled

- **Forward model**: matched filtering against waveform template banks
  (post-Newtonian inspiral + numerical-relativity-calibrated merger and
  ringdown: IMRPhenom, SEOBNR, NRSur families), followed by a hierarchical
  Bayesian model of the compact-binary population.
- **Nuisance parameters**: the entire compact-binary population model —
  mass function shape and location of features, spin distribution, merger
  rate evolution $R(z)$ — plus galaxy catalogue incompleteness for dark
  sirens and detector calibration uncertainty.
- **Inference**: hierarchical Bayesian inference with explicit **selection-effect
  (Malmquist) correction**; stochastic sampling (Bilby, LALInference,
  RIFT). The likelihood is over individual events, not over a compressed
  data vector — structurally unlike every other probe here.
- **Dominant systematics**: the mass–redshift degeneracy, which forces
  cosmology and population modelling to be inferred jointly; waveform
  systematics; calibration.
- **Limiting factor**: event count. This is the one probe on the list that is
  purely statistics-limited today and will improve straightforwardly with
  time.

---

## 7. 21 cm Cosmology and Intensity Mapping

### 7.1 What is observed

The redshifted 21 cm line from the hyperfine spin-flip transition of neutral
hydrogen. Rather than resolving individual galaxies, **intensity mapping**
measures the integrated brightness temperature $T_b$ in coarse voxels,
treating unresolved emission as a continuous field. Because the line has a
known rest frequency, the observed frequency *is* the redshift:
$$\nu_{\rm obs} = \frac{1420\ {\rm MHz}}{1+z}$$

This is the crucial structural advantage: a radio telescope with spectral
resolution automatically delivers a **three-dimensional tomographic map**,
with essentially perfect radial resolution, over enormous volumes.

Two physically distinct regimes:

- **Post-reionisation IM ($z \lesssim 6$)**: HI resides in galaxies; the
  signal is a biased tracer of matter, $T_b \propto \Omega_{\rm HI} b_{\rm HI}$.
- **Reionisation, cosmic dawn and dark ages ($z \sim 6 - 200$)**: the signal
  traces neutral gas against the CMB in absorption or emission, encoding
  astrophysics of the first sources.

### 7.2 Where and when

| | |
|---|---|
| Redshift | $z \sim 0 - 3$ (IM: CHIME, MeerKAT, HIRAX, SKA-MID); $z \sim 6 - 30$ (EoR/cosmic dawn: HERA, LOFAR, MWA, SKA-LOW); $z \gtrsim 30$ (dark ages, future lunar concepts) |
| Scales | $k \sim 0.01 - 1\ h\,{\rm Mpc}^{-1}$ for IM; foreground contamination removes low $k_\parallel$ |
| Regime | Linear to quasi-linear for IM; strongly nonlinear *astrophysically* during the EoR |
| Volume | Potentially the largest cosmological volumes ever surveyed — this is the probe's fundamental promise |

### 7.3 What it constrains

- **BAO and RSD over enormous volumes**, hence $H(z)$, $D_A(z)$, $f\sigma_8$
  at redshifts inaccessible to optical spectroscopy.
- **$f_{\rm NL}$**: the very large volumes and access to ultra-large scales
  make this the leading long-term route to $\sigma(f_{\rm NL}) \lesssim 1$.
- **$\sum m_\nu$, $w_0$–$w_a$**, and the expansion history in a
  spectroscopic-quality 3D survey at low cost per mode.
- **Reionisation and first-star astrophysics** at high $z$: ionising
  efficiency, minimum halo mass for star formation, X-ray heating —
  which is *astrophysics* constrained cosmologically, and worth flagging as
  a different kind of goal.

**Degeneracy note:** the observed amplitude is proportional to the
degenerate product $\Omega_{\rm HI} b_{\rm HI} r$ (with $r$ the
cross-correlation coefficient). Auto-power measurements cannot separate
these, which is why current detections are largely made in
**cross-correlation with optical galaxy surveys** — e.g. CHIME $\times$
eBOSS.

### 7.4 How it is measured

| | |
|---|---|
| Apparatus | Radio interferometers and single-dish arrays with wide fields and dense spectral sampling; transit telescopes with no moving parts (CHIME, HIRAX) |
| Experiments | CHIME, Tianlai, MeerKLASS, uGMRT; HERA, LOFAR, MWA; SKA-Mid and SKA-Low |
| Statistic fitted | 3D power spectrum $P(k_\parallel, k_\perp)$, or cylindrical/spherical bandpowers; cross-power with galaxy surveys |
| Radial resolution | **Exceptional** — set by spectral channel width, effectively spectroscopic for every voxel |
| Noise floor | Thermal (radiometer) noise, integration time, and — dominantly — foreground residuals |

### 7.5 How it is modelled

- **Forward model**: for IM, linear/perturbative $P(k)$ times an HI bias and
  brightness-temperature model, often via an HI halo model or HOD-like
  prescription. For the EoR, semi-numerical simulators (21cmFAST, SimFast21)
  or full radiative-transfer simulations.
- **Nuisance parameters**: $\Omega_{\rm HI}(z)$, $b_{\rm HI}(z)$, shot noise,
  and — overwhelmingly — the **foreground model**. Galactic synchrotron and
  extragalactic point sources are $10^4$–$10^5$ times brighter than the
  signal. Mitigation exploits the fact that foregrounds are spectrally smooth
  while the signal is not: foreground avoidance (working outside the
  "foreground wedge" in $(k_\parallel, k_\perp)$ space) or subtraction (PCA,
  ICA, GPR), both of which remove real signal modes and must be corrected
  for with a transfer function.
- **Inference**: Gaussian likelihood on bandpowers for IM; for the EoR,
  **simulation-based inference** is now standard, since the simulator is
  cheap and the likelihood is intractable.
- **Dominant systematics**: foreground residuals coupled to instrumental
  chromaticity (beam frequency dependence, polarisation leakage), and RFI.
- **Limiting factor**: foreground removal, decisively. The instrument's raw
  sensitivity has not been the bottleneck for a decade.

---

## 8. Comparison Table I — What and Where

| | **Galaxy clustering** | **Weak lensing** | **Primary CMB** | **CMB lensing** | **Ly-$\alpha$ forest** | **Gravitational waves** | **21 cm IM** |
|---|---|---|---|---|---|---|---|
| **Observable field** | Galaxy number density $\delta_g$ | Galaxy ellipticities $\to$ shear $\gamma$ | $\Delta T/T$, $E$-modes | Reconstructed $\kappa$ / $\phi$ | Transmitted flux $F=e^{-\tau}$ | Strain $h(t)$ | Brightness temperature $T_b$ |
| **Traces** | Biased tracer of matter | **Total** matter, projected | Photon–baryon plasma at recombination | **Total** matter, projected | Diffuse IGM (biased) | Metric perturbation; $d_L$ | Neutral hydrogen (biased) |
| **Dimensionality** | 3D | 2D tomographic | 2D single screen | 2D single projection | 1D sightlines $\to$ 3D | Point events (time series) | 3D |
| **Redshift range** | $0 - 2.1$ | $0.2 - 1.5$ (sources) | $z_* = 1090$ | $z_* = 1090$ (kernel $0.5-5$) | $2 - 4.5$ | $0 - 1$ (LVK); $0-10$ (3G) | $0-3$; $6-30$ |
| **Effective redshift** | $0.3 - 1.3$ | $\sim 0.3 - 0.6$ | 1090 | $\sim 2$ | $\sim 3$ | $\lesssim 0.1$ today | $\sim 1$; $\sim 8$ |
| **Scales ($k$, $h$/Mpc)** | $0.005 - 0.25$ | $0.1 - 5$ | — (linear, $\ell$-space) | $0.05 - 0.5$ | $0.1 - 10$ | — | $0.01 - 1$ |
| **Scales ($\ell$ or $L$)** | — | $100 - 3000$ | $2 - 4000$ | $40 - 1300$ | — | — | — |
| **Regime** | Linear / mildly nonlinear | **Deeply nonlinear** | **Linear** | Linear / quasi-linear | Mildly nonlinear + thermal | Strong-field source, linear propagation | Linear / quasi-linear |
| **Radial resolution** | Spectroscopic ($10^{-4}$) | Photometric ($0.05(1+z)$) | None | None | Excellent along sightline | None intrinsically | Excellent (spectral) |

---

## 9. Comparison Table II — What It Tells Us

| | **Galaxy clustering** | **Weak lensing** | **Primary CMB** | **CMB lensing** | **Ly-$\alpha$ forest** | **Gravitational waves** | **21 cm IM** |
|---|---|---|---|---|---|---|---|
| **Geometry / growth** | **Both** (BAO / RSD) | Growth (+ some geometry) | Geometry at $z_*$ + early physics | Growth | Growth (amplitude, shape) | **Pure geometry** | Both |
| **Best-constrained combination** | $D_M/r_d$, $D_H/r_d$, $f\sigma_8$, $b_1\sigma_8$ | $S_8=\sigma_8(\Omega_m/0.3)^{0.5}$ | $\theta_*$, $\Omega_b h^2$, $\Omega_c h^2$, $A_s e^{-2\tau}$, $n_s$ | $\sigma_8(\Omega_m/0.3)^{0.25}$ | $\Delta^2_L$, $n_{\rm eff}$ at small $k$ | $H_0$ (absolute) | $\Omega_{\rm HI}b_{\rm HI}$, $D_A$, $H$, $f\sigma_8$ |
| **Primary parameters** | $\Omega_m$, $H_0 r_d$, $\sigma_8$, $n_s$ | $\sigma_8$, $\Omega_m$ | Full 6-parameter $\Lambda$CDM | $\sigma_8$, $\Omega_m$ | $A_s$, $n_s$, small-scale power | $H_0$ | $\Omega_m$, $H(z)$, $f\sigma_8$ |
| **$\sum m_\nu$** | Good | Moderate | Weak (via lensing) | **Excellent** | **Excellent** | No | Good |
| **$w_0$–$w_a$** | **Excellent** | Good | Weak alone (needs late-time anchor) | Moderate | Weak | Good (3G) | Good |
| **$f_{\rm NL}$** | Good (scale-dependent bias) | Weak | Good (bispectrum) | Weak | Weak | No | **Excellent (future)** |
| **Modified gravity** | $\mu$ (growth) | $\Sigma$ (lensing potential) | Limited | $\Sigma$ | Limited | **$d_L^{\rm GW}/d_L^{\rm EM}$, $c_{\rm GW}$ — unique** | $\mu$ |
| **Calibration** | Relative (needs $r_d$) | Relative | **Absolute** ($r_s$ computed) | Relative | Relative | **Absolute** (waveform) | Relative |
| **Current headline precision** | $\sim 0.5\%$ on BAO distances | $\sim 2\%$ on $S_8$ | $0.03\%$ on $\theta_*$ | $\sim 1.7\%$ on lensing amplitude | Percent-level on $P_{\rm 1D}$ | $\sim 15\%$ on $H_0$ | Detections in cross-correlation |

---

## 10. Comparison Table III — How We Model It

| | **Galaxy clustering** | **Weak lensing** | **Primary CMB** | **CMB lensing** | **Ly-$\alpha$ forest** | **Gravitational waves** | **21 cm IM** |
|---|---|---|---|---|---|---|---|
| **Forward model** | Boltzmann + 1-loop EFT/TNS | Boltzmann + nonlinear emulator + baryons + Limber | **Boltzmann only** | Boltzmann + mild nonlinear + Limber | **Hydrodynamic simulations + emulator** | NR-calibrated waveforms + population model | Boltzmann + HI bias model; or 21cmFAST |
| **Key nuisances** | $b_1,b_2,b_{s^2}$, counterterms, shot noise | IA, shear bias $m$, photo-$z$, baryonic feedback | Foregrounds, beams, calibration | $N^{(0)}/N^{(1)}$ biases, tSZ/CIB contamination | $T_0$, $\gamma$, $\bar F$, DLAs, metals, continuum | Mass function, spin, $R(z)$, catalogue completeness | **Foregrounds**, $\Omega_{\rm HI}b_{\rm HI}$, beam chromaticity |
| **Approx. nuisance count** | 7–10 per tracer per bin | 10–20 | ~20 (instrumental only) | **Very few** | Often $>$ cosmological params | ~10 population params | Effectively continuous (foreground modes) |
| **Likelihood / inference** | Gaussian + mock covariance | Gaussian + analytic covariance; nested sampling | Gaussian / Hamimeche–Lewis; MCMC | Gaussian bandpowers, sim-based debiasing | Emulator likelihood; SBI | **Hierarchical Bayesian with selection effects** | Gaussian bandpowers; **SBI** for EoR |
| **Dominant systematic** | Perturbative model at $k_{\rm max}$; target selection | Intrinsic alignments, baryons, photo-$z$ | Foreground separation; $\tau$ | Foreground bias in reconstruction | IGM thermal state | Mass–redshift degeneracy; waveforms | **Foreground residuals** |
| **Limited by** | Theory | Astrophysical modelling | **Cosmic variance** | Detector noise | Simulation accuracy | **Event count (statistics)** | Foreground removal |
| **Model maturity** | High | Moderate | **Very high** | High | Moderate | Moderate | Early |

---

## 11. Synthesis: why we combine them

Four points worth making explicitly at the end of a lecture:

**1. Degeneracy directions are complementary, not merely additive.**
Cosmic shear measures $\sigma_8 \Omega_m^{0.5}$; CMB lensing measures
$\sigma_8 \Omega_m^{0.25}$; BAO measures $\Omega_m$ nearly independently of
amplitude. The intersection of three differently oriented bananas is far
smaller than any single one, and the gain is geometric rather than
statistical.

**2. Independent absolute calibrations test the model, not just the parameters.**
The CMB sound horizon and the GW waveform amplitude are two physically
unrelated absolute rulers. If they disagree on $H_0$, no amount of data
volume in a single probe will resolve it — the disagreement is the result.

**3. Geometry versus growth is a null test of general relativity.**
$\Lambda$CDM + GR uniquely predicts the growth history given the expansion
history. Measuring $f\sigma_8(z)$ from RSD and $D_A(z)$ from BAO in the same
survey turns a parameter measurement into a consistency test.

**4. Systematics are almost entirely uncorrelated across probes.**
Intrinsic alignments have nothing to do with IGM thermal history, which has
nothing to do with Galactic synchrotron foregrounds, which has nothing to do
with compact-binary selection effects. Agreement between probes is therefore
strong evidence; disagreement localises a problem. This is the deepest
argument for the multi-probe programme, and it is worth stating as such.

---

## Appendix — Suggested figures for the lecture

1. The lensing efficiency kernel $W(\chi)$ for a few tomographic bins,
   overlaid with the CMB lensing kernel — makes the "different effective
   redshift" point visually obvious.
2. $\Omega_m$–$\sigma_8$ contours for cosmic shear, CMB lensing, clustering
   and primary CMB on one axis, showing the rotated degeneracies.
3. A single $k$-axis from $10^{-4}$ to $10\ h\,{\rm Mpc}^{-1}$ with coloured
   bars marking the range each probe covers — the clearest single summary of
   this entire document.
4. A $d_L(z)$ diagram with SNe and standard sirens, illustrating absolute
   versus relative calibration.
5. The $(k_\parallel, k_\perp)$ foreground wedge for 21 cm, to make concrete
   why the modes are lost rather than merely noisy.
