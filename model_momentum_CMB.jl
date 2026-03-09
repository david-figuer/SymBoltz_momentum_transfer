# αCDM in SymBoltz
# References:
#    https://inspirehep.net/literature/1849615
#    https://inspirehep.net/literature/1869592
#    https://inspirehep.net/literature/2156882
#    https://inspirehep.net/literature/2615535
#    https://inspirehep.net/literature/2765256
#    https://inspirehep.net/literature/2842645
#    Doctoral thesis: https://inspirehep.net/literature/2722259
# The model was first introduced in:
#    M. Asghari, J. Beltrán Jiménez, S. Khosravi & D. F. Mota,
#    On structure formation from a small-scales-interacting dark sector,
#    arXiv:1902.05532.

# This .jl serves for computing the CMB TT power spectrum vs alpha using ONLY SymBoltz

# Start Julia with: julia -tauto
# Then run: include("______.jl")


#Necessary usings
using SymBoltz
using Plots
using LinearAlgebra: BLAS
using Unitful
using UnitfulAstro

# ============================================================
# USER CONFIG
# ============================================================
# output paths
const OUT_TT = "/mnt/wwn-0x50014ee2c101e66c-part1/Zona_de_Trabajo/Proyecto_SymBoltz/Symboltz_vs_CLASS/TT_SymBoltz_modelo.png"
const OUT_EE = "/mnt/wwn-0x50014ee2c101e66c-part1/Zona_de_Trabajo/Proyecto_SymBoltz/Symboltz_vs_CLASS/EE_SymBoltz_modelo.png"
const OUT_TE = "/mnt/wwn-0x50014ee2c101e66c-part1/Zona_de_Trabajo/Proyecto_SymBoltz/Symboltz_vs_CLASS/TE_SymBoltz_modelo.png"

# Cosmology params, general parameters. Models paramters to be defined later
h      = 0.70
Ob     = 0.05
Ocdm   = 0.30
YHe    = 0.25
T_cmb  = 2.7255
Neff   = 3.046
m_nu_eV = 0.02
w0     = -0.98
wa     = 0.0
cs2_fld = 1.0
A_s    = 2e-9
n_s    = 0.94

# Run these alphas (ONLY SymBoltz)
alphas = [0.0, 1.0, 10.0, 100.0]

# TT grid
jl = SphericalBesselCache(2:25:2502)
ls = 2:2502
modes = [:TT,:TE,:EE]


# ============================================================
# SYMBOLIC MODEL DEFINITION (your αCDM with α)
# ============================================================

# Construct a symbolic αCDM model from SymBoltz docs with w0 component (fix wa=0 needed) 
function αCDM(; lγmax=6, lνmax=6, lhmax=6, name=:w₀wₐCDM, kwargs...)
    @unpack kB, h, ħ, c, GN, H100, eV, me, mH, mHe, σT, aR, δkron, smoothifelse = SymBoltz
    @unpack λH2s1s, EH2s1s, EH∞2s, EHe2s1s, λHe2p1s, fHe2p1s, EHe2p2s, EHe∞2s, EHe⁺∞1s, EHet∞2s, EHet2s1s, EHet2p2s, λHet2p1s, fHet2p1s = SymBoltz  # energy levels

         ϵ = 1e-9
    ΛH = 8.2245809
    ΛHe = 51.3
    A2ps = 1.798287e9
    A2pt = 177.58e0
    αHfit(T; F=1.125, a=4.309, b=-0.6166, c=0.6703, d=0.5300, T₀=1e4) = F * 1e-19 * a * (T/T₀)^b / (1 + c * (T/T₀)^d)
    αHefit(T; q=NaN, p=NaN, T1=10^5.114, T2=3.0) = q / (√(T/T2) * (1+√(T/T2))^(1-p) * (1+√(T/T1))^(1+p))
    KHfitfactorfunc(a, A, z, w) = A*exp(-((log(a)+z)/w)^2)
    γHe(; A=NaN, σ=NaN, f=NaN) = 3*A*fHe*(1-XHe⁺+ϵ)*c^2 / (8π*σ*√(2π/(β*mHe*c^2))*(1-XH⁺+ϵ)*f^3)
    # massive neutrino distribution function and quadrature momenta
    nx = 4
    f₀(x) = 1 / (exp(x) + 1)
    dlnf₀_dlnx(x) = -x / (1 + exp(-x))
    x, W = SymBoltz.momentum_quadrature(f₀, nx)
    x² = x .^ 2
    ∫dx_x²_f₀(f) = sum(collect(f .* W))

    @independent_variables τ # conformal time
    D = Differential(τ) # derivative operator

    pars = @parameters begin
        k, τ0, # wavenumber and conformal time today
        h, # reduced Hubble parameter (overwrites Planck constant above!)
        Ωc0, # cold dark matter
        Ωb0, YHe, fHe, κ0, # baryons and recombination
        Tγ0, Ωγ0, # photons
        Ων0, Tν0, Neff, # massless neutrinos
        mh, mh_eV, Nh, Th0, Ωh0, yh0, Iρh0, # massive neutrinos
        ΩX0, w0, wa, cXs2, # dark energy
        zre1, Δzre1, nre1, # 1st reionization
        zre2, Δzre2, nre2, # 2nd reionization
        C, # integration constant in initial conditions
        As, ns # primordial power spectrum
        α      # !momentum
    end

    vars = @variables begin
        a(τ), z(τ), ℋ(τ), H(τ), Ψ(τ,k), Φ(τ,k), χ(τ), # metric
        ρ(τ), P(τ), δρ(τ,k), Π(τ,k), # gravity
        ρb(τ), Tb(τ), δb(τ,k), Δb(τ,k), θb(τ,k), # baryons
        κ(τ), κ̇(τ), _κ(τ), v(τ), csb2(τ), β(τ), ΔT(τ), DTb(τ), DTγ(τ), μc²(τ), Xe(τ), nH(τ), nHe(τ), ne(τ), Xe(τ), ne(τ), λe(τ), Hrec(τ), # recombination
        XH⁺(τ), nH(τ), αH(τ), βH(τ), KH(τ), KHfitfactor(τ), CH(τ) # Hydrogen recombination
        nHe(τ), XHe⁺(τ), XHe⁺⁺(τ), αHe(τ), βHe(τ), RHe⁺(τ), τHe(τ), KHe(τ), invKHe0(τ), invKHe1(τ), invKHe2(τ), CHe(τ), DXHe⁺(τ), DXHet⁺(τ), γ2ps(τ), αHet(τ), βHet(τ), τHet(τ), pHet(τ), CHet(τ), CHetnum(τ), γ2pt(τ), # Helium recombination
        Xre1(τ), Xre2(τ), # reionization
        ργ(τ), Pγ(τ), wγ(τ), Tγ(τ), Fγ0(τ,k), Fγ(τ,k)[1:lγmax], Gγ0(τ,k), Gγ(τ,k)[1:lγmax], δγ(τ,k), θγ(τ,k), σγ(τ,k), Πγ(τ,k) # photons
        ρc(τ), δc(τ,k), Δc(τ,k), θc(τ,k) # cold dark matter
        ρν(τ), Pν(τ), wν(τ), Tν(τ), Fν0(τ,k), Fν(τ,k)[1:lνmax], δν(τ,k), θν(τ,k), σν(τ,k), # massless neutrinos
        ρh(τ), Ph(τ), wh(τ), Ωh(τ), Th(τ), yh(τ), csh2(τ,k), δh(τ,k), Δh(τ,k), σh(τ,k), uh(τ,k), θh(τ,k), Eh(τ)[1:nx], ψh0(τ,k)[1:nx], ψh(τ,k)[1:nx,1:lhmax], Iρh(τ), IPh(τ), Iδρh(τ,k), # massive neutrinos
        ρX(τ), PX(τ), wX(τ), ẇX(τ), cXa2(τ), δX(τ,k), θX(τ,k), ΔX(τ,k), # dark energy
        fν(τ) # misc
        Δm(τ,k) # matter source functions
        ST_SW(τ,k), ST_ISW(τ,k), ST_Doppler(τ,k), ST_polarization(τ,k), ST(τ,k), SE_kχ²(τ,k), Sψ(τ,k) # CMB source functions
        Γα(τ,k), Rα(τ,k), momentum_eq_DM(τ,k), momentum_eq_DE(τ,k)  # !momentum
    end

    eqs = [
        # metric equations
        z ~ 1/a - 1
        ℋ ~ D(a) / a
        H ~ ℋ / a
        χ ~ τ0 - τ
        # gravity equations
        D(a) ~ √(8*Num(π)/3 * ρ) * a^2                       # 1st Friedmann equation
        D(Φ) ~ -4*Num(π)/3*a^2/ℋ*δρ - k^2/(3*ℋ)*Φ - ℋ*Ψ
        k^2 * (Φ - Ψ) ~ 12*Num(π) * a^2 * Π
        ρ ~ ρc + ρb + ργ + ρν + ρh + ρX
        P ~ Pγ + Pν + Ph + PX
        δρ ~ δc*ρc + δb*ρb + δγ*ργ + δν*ρν + δh*ρh + δX*ρX
        Π ~ (1+wγ)*ργ*σγ + (1+wν)*ρν*σν + (1+wh)*ρh*σh
        # baryon recombination
        β ~ 1 / (kB*Tb)
        λe ~ 2π*ħ / √(2π*me/β)
        Hrec ~ H100 * h * H
        D(_κ) ~ -a/(H100*h) * ne * σT * c
        κ̇ ~ D(_κ)
        κ ~ _κ - κ0
        v ~ D(exp(-κ)) |> expand_derivatives
        csb2 ~ kB/μc² * (Tb - D(Tb)/3ℋ)
        μc² ~ mH*c^2 / (1 + (mH/mHe-1)*YHe + Xe*(1-YHe))
        DTb ~ -2*Tb*ℋ - a/h * 8/3*σT*aR/H100*Tγ^4 / (me*c) * Xe / (1+fHe+Xe) * ΔT
        DTγ ~ D(Tγ)
        D(ΔT) ~ DTb - DTγ
        Tb ~ ΔT + Tγ
        nH ~ (1-YHe) * ρb*(H100*h)^2/GN / mH
        nHe ~ fHe * nH
        ne ~ Xe * nH
        Xe ~ 1*XH⁺ + fHe*XHe⁺ + XHe⁺⁺ + Xre1 + Xre2
        # baryon H⁺ + e⁻ recombination
        αH ~ αHfit(Tb)
        βH ~ αH / λe^3 * exp(-β*EH∞2s)
        KHfitfactor ~ 1 + KHfitfactorfunc(a, -0.14, 7.28, 0.18) + KHfitfactorfunc(a, 0.079, 6.73, 0.33)
        KH ~ KHfitfactor/8π * λH2s1s^3 / Hrec
        CH ~ smoothifelse(XH⁺ - 0.99, (1 + KH*ΛH*nH*(1-XH⁺)) / (1 + KH*(ΛH+βH)*nH*(1-XH⁺)), 1; k = 1e3)
        D(XH⁺) ~ -a/(H100*h) * CH * (αH*XH⁺*ne - βH*(1-XH⁺)*exp(-β*EH2s1s))
        # baryon He⁺ + e⁻ singlet recombination
        αHe ~ αHefit(Tb; q=10^(-16.744), p=0.711)
        βHe ~ 4 * αHe / λe^3 * exp(-β*EHe∞2s)
        KHe ~ 1 / (invKHe0 + invKHe1 + invKHe2)
        invKHe0 ~ 8π*Hrec / λHe2p1s^3
        τHe ~ 3*A2ps*nHe*(1-XHe⁺+ϵ) / invKHe0
        invKHe1 ~ -exp(-τHe) * invKHe0
        γ2ps ~ γHe(A = A2ps, σ = 1.436289e-22, f = fHe2p1s)
        invKHe2 ~ A2ps/(1+0.36*γ2ps^0.86)*3*nHe*(1-XHe⁺)
        CHe ~ smoothifelse(XHe⁺ - 0.99, (exp(-β*EHe2p2s) + KHe*ΛHe*nHe*(1-XHe⁺)) / (exp(-β*EHe2p2s) + KHe*(ΛHe+βHe)*nHe*(1-XHe⁺)), 1; k = 1e3)
        DXHe⁺ ~ -a/(H100*h) * CHe * (αHe*XHe⁺*ne - βHe*(1-XHe⁺)*exp(-β*EHe2s1s))
        # baryon He⁺ + e⁻ triplet recombination
        αHet ~ αHefit(Tb; q=10^(-16.306), p=0.761)
        βHet ~ 4/3 * αHet / λe^3 * exp(-β*EHet∞2s)
        τHet ~ A2pt*nHe*(1-XHe⁺+ϵ)*3 * λHet2p1s^3/(8π*Hrec)
        pHet ~ (1 - exp(-τHet)) / τHet
        γ2pt ~ γHe(A = A2pt, σ = 1.484872e-22, f = fHet2p1s)
        CHetnum ~ A2pt*(pHet+1/(1+0.66*γ2pt^0.9)/3)*exp(-β*EHet2p2s)
        CHet ~ (ϵ + CHetnum) / (ϵ + CHetnum + βHet)
        DXHet⁺ ~ -a/(H100*h) * CHet * (αHet*XHe⁺*ne - βHet*(1-XHe⁺)*3*exp(-β*EHet2s1s))
        # baryon He⁺ + e⁻ total recombination
        D(XHe⁺) ~ DXHe⁺ + DXHet⁺
        # baryon He⁺⁺ + e⁻ recombination
        RHe⁺ ~ 1 * exp(-β*EHe⁺∞1s) / (nH * λe^3)
        XHe⁺⁺ ~ 2*RHe⁺*fHe / (1+fHe+RHe⁺) / (1 + √(1 + 4*RHe⁺*fHe/(1+fHe+RHe⁺)^2))
        # reionization
        Xre1 ~ smoothifelse((1+zre1)^nre1 - (1+z)^nre1, 0, 1 + fHe; k = 1/(nre1*(1+zre1)^(nre1-1)*Δzre1))
        Xre2 ~ smoothifelse((1+zre2)^nre2 - (1+z)^nre2, 0, 0 + fHe; k = 1/(nre2*(1+zre2)^(nre2-1)*Δzre2))
        # baryons
        ρb ~ 3/(8*Num(π)) * Ωb0 / a^3
        D(δb) ~ -θb - 3*ℋ*csb2*δb + 3*D(Φ)
        D(θb) ~ -ℋ*θb + k^2*csb2*δb + k^2*Ψ - 4//3*κ̇*ργ/ρb*(θγ-θb)
        Δb ~ δb + 3ℋ*θb/k^2
        # photons
        Tγ ~ Tγ0 / a
        ργ ~ 3/(8*Num(π)) * Ωγ0 / a^4
        wγ ~ 1//3
        Pγ ~ wγ * ργ
        D(Fγ0) ~ -k*Fγ[1] + 4*D(Φ)
        D(Fγ[1]) ~ k/3*(Fγ0-2*Fγ[2]+4*Ψ) - 4//3 * κ̇/k * (θb - θγ)
        [D(Fγ[l]) ~ k/(2l+1) * (l*Fγ[l-1] - (l+1)*Fγ[l+1]) + κ̇ * (Fγ[l] - δkron(l,2)//10*Πγ) for l in 2:lγmax-1]...
        D(Fγ[lγmax]) ~ k*Fγ[lγmax-1] - (lγmax+1) / τ * Fγ[lγmax] + κ̇ * Fγ[lγmax]
        δγ ~ Fγ0
        θγ ~ 3*k*Fγ[1]/4
        σγ ~ Fγ[2]/2
        Πγ ~ Fγ[2] + Gγ0 + Gγ[2]
        D(Gγ0) ~ k * (-Gγ[1]) + κ̇ * (Gγ0 - Πγ/2)
        D(Gγ[1]) ~ k/(2*1+1) * (1*Gγ0 - 2*Gγ[2]) + κ̇ * Gγ[1]
        [D(Gγ[l]) ~ k/(2l+1) * (l*Gγ[l-1] - (l+1)*Gγ[l+1]) + κ̇ * (Gγ[l] - δkron(l,2)//10*Πγ) for l in 2:lγmax-1]...
        D(Gγ[lγmax]) ~ k*Gγ[lγmax-1] - (lγmax+1) / τ * Gγ[lγmax]
        # cold dark matter
        ρc ~ 3/(8*Num(π)) * Ωc0 / a^3
        D(δc) ~ -(θc-3*D(Φ))
        D(θc) ~ -ℋ*θc + k^2*Ψ + momentum_eq_DM   # !momentum
        Δc ~ δc + 3ℋ*θc/k^2
        # massless neutrinos
        ρν ~ 3/(8*Num(π)) * Ων0 / a^4
        wν ~ 1//3
        Pν ~ wν * ρν
        Tν ~ Tν0 / a
        D(Fν0) ~ -k*Fν[1] + 4*D(Φ)
        D(Fν[1]) ~ k/3*(Fν0-2*Fν[2]+4*Ψ)
        [D(Fν[l]) ~ k/(2*l+1) * (l*Fν[l-1] - (l+1)*Fν[l+1]) for l in 2:lνmax-1]...
        D(Fν[lνmax]) ~ k*Fν[lνmax-1] - (lνmax+1) / τ * Fν[lνmax]
        δν ~ Fν0
        θν ~ 3*k*Fν[1]/4
        σν ~ Fν[2]/2
        # massive neutrinos
        Th ~ Th0 / a
        yh ~ yh0 * a
        Iρh ~ ∫dx_x²_f₀(Eh)
        IPh ~ ∫dx_x²_f₀(x² ./ Eh)
        ρh ~ 2Nh/(2*π^2) * (kB*Th)^4 / (ħ*c)^3 * Iρh / ((H100*h*c)^2/GN)
        Ph ~ 2Nh/(6*π^2) * (kB*Th)^4 / (ħ*c)^3 * IPh / ((H100*h*c)^2/GN)
        wh ~ Ph / ρh
        Iδρh ~ ∫dx_x²_f₀(Eh .* ψh0)
        δh ~ Iδρh / Iρh
        Δh ~ δh + 3ℋ*(1+wh)*θh/k^2
        uh ~ ∫dx_x²_f₀(x .* ψh[:,1]) / (Iρh + IPh/3)
        θh ~ k * uh
        σh ~ (2//3) * ∫dx_x²_f₀(x² ./ Eh .* ψh[:,2]) / (Iρh + IPh/3)
        csh2 ~ ∫dx_x²_f₀(x² ./ Eh .* ψh0) / Iδρh
        [Eh[i] ~ √(x[i]^2 + yh^2) for i in 1:nx]...
        [D(ψh0[i]) ~ -k * x[i]/Eh[i] * ψh[i,1] - D(Φ) * dlnf₀_dlnx(x[i]) for i in 1:nx]...
        [D(ψh[i,1]) ~ k/3 * x[i]/Eh[i] * (ψh0[i] - 2*ψh[i,2]) - k/3 * Eh[i]/x[i] * Ψ * dlnf₀_dlnx(x[i]) for i in 1:nx]...
        [D(ψh[i,l]) ~ k/(2*l+1) * x[i]/Eh[i] * (l*ψh[i,l-1] - (l+1) * ψh[i,l+1]) for i in 1:nx, l in 2:lhmax-1]...
        [D(ψh[i,lhmax]) ~ k/(2*lhmax+1) * x[i]/Eh[i] * (lhmax*ψh[i,lhmax-1] - (lhmax+1) * ((2*lhmax+1) * Eh[i]/x[i] * ψh[i,lhmax] / (k*τ) - ψh[i,lhmax-1])) for i in 1:nx]...
        # dark energy
        wX ~ w0 + wa * (1 - a)
        ẇX ~ D(wX)
        ρX ~ 3/(8*Num(π))*ΩX0 * abs(a)^(-3*(1+w0+wa)) * exp(-3wa*(1-a))
        PX ~ wX * ρX
        cXa2 ~ wX - ẇX/(3ℋ*(1+wX))
        D(δX) ~ -(1+wX)*(θX-3*D(Φ)) - 3ℋ*(cXs2-wX)*δX - 9*(ℋ/k)^2*(1+wX)*(cXs2-cXa2)*θX
        D(θX) ~ -ℋ*(1-3*cXs2)*θX + cXs2/(1+wX)*k^2*δX + k^2*Ψ - momentum_eq_DE  # !momentum
        ΔX ~ δX + 3ℋ*(1+wX)*θX/k^2
        # misc
        fν ~ (ρν + ρh) / (ρν + ρh + ργ)
        # matter source functions
        Δm ~ (ρb*Δb + ρc*Δc + ρh*Δh) / (ρb + ρc + ρh)
        # CMB source functions
        ST_SW ~ v * (δγ/4 + Ψ + Πγ/16)
        ST_ISW ~ exp(-κ) * D(Ψ + Φ) |> expand_derivatives
        ST_Doppler ~ D(v*θb) / k^2 |> expand_derivatives
        ST_polarization ~ 3/(16*k^2) * D(D(v*Πγ)) |> expand_derivatives
        ST ~ ST_SW + ST_ISW + ST_Doppler + ST_polarization
        SE_kχ² ~ 3/16 * v*Πγ
        Sψ ~ 0
        # !momentum
        Γα ~ α * ( a / ρc) * 3/(8*Num(π))    # !momentum
        Rα ~ ρc / ( (1+wX) * ρX )            # !momentum
        momentum_eq_DM ~ Γα * (θX-θc)        # !momentum
        momentum_eq_DE ~ Γα * Rα * (θX-θc)   # !momentum
    ]

    initialization_eqs = [
            # metric/gravity
        Ψ ~ 20C / (15 + 4fν)
        D(a) ~ a / τ
        # baryons
        δb ~ -3//2 * Ψ
        θb ~ 1//2 * (k^2*τ) * Ψ
        # photons
        Fγ0 ~ -2*Ψ
        Fγ[1] ~ 2//3 * k*τ*Ψ
        Fγ[2] ~ -8//15 * k/κ̇ * Fγ[1]
        [Fγ[l] ~ -l//(2*l+1) * k/κ̇ * Fγ[l-1] for l in 3:lγmax]...
        Gγ0 ~ 5//16 * Fγ[2]
        Gγ[1] ~ -1//16 * k/κ̇ * Fγ[2]
        Gγ[2] ~ 1//16 * Fγ[2]
        [Gγ[l] ~ -l//(2l+1) * k/κ̇ * Gγ[l-1] for l in 3:lγmax]...
        # cold dark matter
        δc ~ -3//2 * Ψ
        θc ~ 1//2 * (k^2*τ) * Ψ
        # massless neutrinos
        δν ~ -2 * Ψ
        θν ~ 1//2 * (k^2*τ) * Ψ
        σν ~ 1//15 * (k*τ)^2 * Ψ
        [Fν[l] ~ +l//(2*l+1) * k*τ * Fν[l-1] for l in 3:lνmax]...
        # massless neutrinos
        [ψh0[i] ~ -1//4 * (-2*Ψ) * dlnf₀_dlnx(x[i]) for i in 1:nx]...
        [ψh[i,1] ~ -1//3 * Eh[i]/x[i] * (1/2*k*τ*Ψ) * dlnf₀_dlnx(x[i]) for i in 1:nx]...
        [ψh[i,2] ~ -1//2 * (1//15*(k*τ)^2*Ψ) * dlnf₀_dlnx(x[i]) for i in 1:nx]...
        [ψh[i,l] ~ 0 for i in 1:nx, l in 3:lhmax]...
        # dark energy
        δX ~ -3//2 * (1+wX) * Ψ
        θX ~ 1//2 * (k^2*τ) * Ψ
    ]

    initial_conditions = [
        τ0 => NaN
        C => 1//2
        XHe⁺ => 1.0
        XH⁺ => 1.0
        _κ => 0.0
        κ0 => NaN
        ΔT => 0.0

        zre1 => 7.6711
        Δzre1 => 0.5
        nre1 => 3/2
        zre2 => 3.5
        Δzre2 => 0.5
        nre2 => 1

        Tν0 => (4/11)^(1/3) * Tγ0
        Ων0 => Neff * 7/8 * (4/11)^(4/3) * Ωγ0

        Nh => 3
        Th0 => (4/11)^(1/3) * Tγ0
        ΩX0 => 1 - Ωγ0 - Ωc0 - Ωb0

        Ωγ0 => π^2/15 * (kB*Tγ0)^4 / (ħ^3*c^5) * 8π*GN / (3*(H100*h)^2)

        mh => mh_eV * eV/c^2
        yh0 => mh*c^2 / (kB*Th0)
        Iρh0 => ∫dx_x²_f₀(@. √(x^2 + yh0^2))
        Ωh0 => Nh * 8*Num(π)/3 * 2/(2*Num(π)^2) * (kB*Th0)^4 / (ħ*c)^3 * Iρh0 / ((H100*h*c)^2/GN)

        fHe => YHe / (mHe/mH*(1-YHe))
    ]

    guesses = [ a => τ ]

    return complete(System(eqs, τ, vars, pars; initialization_eqs, initial_conditions, guesses, name, kwargs...))
end

# ============================================================
# MAIN
# ============================================================

function main()
    BLAS.set_num_threads(1)
#We call here the αCDM model
    M = αCDM()
    #Fix parameters according to previous definitions
    base_p = Dict(
        M.h     => h,
        M.Ωc0   => Ocdm,
        M.Ωb0   => Ob,
        M.YHe   => YHe,
        M.Tγ0   => T_cmb,
        M.Neff  => Neff,
        M.mh_eV => m_nu_eV,
        M.w0    => w0,
        M.wa    => wa,
        M.cXs2  => cs2_fld,
        M.As    => A_s,
        M.ns    => n_s,
    )

    sym_tt = Dict{Float64, Vector{Float64}}()
    sym_ee = Dict{Float64, Vector{Float64}}()
    sym_te = Dict{Float64, Vector{Float64}}()

    # colors per alpha: grab first N distinct colors from Plots palette
    pal = palette(:auto)
    colors = pal[1:length(alphas)]

    # =============================
    # Run loop over alphas
    # =============================
    for (i, αval) in enumerate(alphas)
        # SymBoltz
        p = copy(base_p)
        p[M.α] = αval
        prob = CosmologyProblem(M, p)

        Dls = spectrum_cmb(
            modes, prob, jl, ls;
            ptopts = (alg = SymBoltz.Rodas5P(linsolve = SymBoltz.RFLUFactorization()), reltol = 1e-4, abstol = 1e-4),
            sourceopts = (refine = false,),
            normalization = :Dl,
            kτ0s = 0.05*jl.l[begin]:2π/2:1.8*jl.l[end],
            coarse_length = 300,
            verbose = true
        )

        sym_tt[αval] = Float64.(Dls[:,1] .* 1e12)
                println("Loaded TT from SymBoltz for alpha=", αval)
        sym_ee[αval] = Float64.(Dls[:,3] .* 1e12)
                println("Loaded EE from SymBoltz for alpha=", αval)
        sym_te[αval] = Float64.(Dls[:,2] .* 1e12)
                println("Loaded TE from SymBoltz for alpha=", αval)

        prob = nothing
        p = nothing
        GC.gc()

    end

    # =============================
    # Plot TT: same color per alpha
    # =============================
    pltTT = plot(
            xlabel="ℓ",
            ylabel="10¹² D(ℓ) (TT)",
            legend=:bottomleft
    )

    for (i, αval) in enumerate(alphas)
            col = colors[i]
            plot!(pltTT, log10.(ls), log10.(sym_tt[αval]),
                  label="SymBoltz α=$(αval)",
                  linestyle=:solid,
                  color=col)
    end

    savefig(pltTT, OUT_TT)
    println("Saved TT-plot to: ", OUT_TT)

    pltTT = nothing
    GC.gc()
    
    # =============================
    # Plot EE: same color per alpha
    # =============================
    pltEE = plot(
            xlabel="ℓ",
            ylabel="10¹² D(ℓ) (EE)",
            legend=:bottomleft
    )

    for (i, αval) in enumerate(alphas)
            col = colors[i]
            plot!(pltEE, ls, sym_ee[αval],
                  label="SymBoltz α=$(αval)",
                  linestyle=:solid,
                  color=col)
    end

    savefig(pltEE, OUT_EE)
    println("Saved EE-plot to: ", OUT_EE)

    pltTT = nothing
    GC.gc()
    
    
    # =============================
    # Plot TE: same color per alpha
    # =============================
    pltTE = plot(
            xlabel="ℓ",
            ylabel="10¹² D(ℓ) (TE)",
            legend=:bottomleft
    )

    for (i, αval) in enumerate(alphas)
            col = colors[i]
            plot!(pltTE, ls, sym_te[αval],
                  label="SymBoltz α=$(αval)",
                  linestyle=:solid,
                  color=col)
    end

    savefig(pltTE, OUT_TE)
    println("Saved TE-plot to: ", OUT_TE)

    pltTT = nothing
    GC.gc()

    closeall()
    GC.gc()
    GC.gc()
end

main()
GC.gc()
GC.gc()
