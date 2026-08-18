# B6 aggregate report

units loaded: 16800 seed-records | datasets: ['ImageNet16-120', 'cifar10-valid', 'cifar100'] | algos: ['ls', 're', 'reinforce', 'rs'] | spaces: ['full', 'naswot50', 'no_none', 'param50', 'randM4096', 'randM7813', 'synflow50']

## Mean final test accuracy (over seeds)

| dataset | algo | full | naswot50 | no_none | param50 | randM4096 | randM7813 | synflow50 |
|---|---|---|---|---|---|---|---|---|
| ImageNet16-120 | ls | 46.32 | 46.33 | 46.31 | 46.38 | 46.35 | 46.41 | 46.37 |
| ImageNet16-120 | re | 46.43 | 46.42 | 46.45 | 46.41 | 46.33 | 46.36 | 46.39 |
| ImageNet16-120 | reinforce | 46.37 | 46.44 | 46.36 | 46.34 | 46.27 | 46.24 | 46.35 |
| ImageNet16-120 | rs | 46.07 | 46.19 | 46.23 | 46.23 | 46.16 | 46.05 | 46.25 |
| cifar10-valid | ls | 91.10 | 91.10 | 91.09 | 91.10 | 91.16 | 91.15 | 91.10 |
| cifar10-valid | re | 91.14 | 91.14 | 91.12 | 91.13 | 91.15 | 91.16 | 91.14 |
| cifar10-valid | reinforce | 91.16 | 91.17 | 91.12 | 91.17 | 91.15 | 91.17 | 91.16 |
| cifar10-valid | rs | 91.05 | 91.07 | 91.13 | 91.11 | 91.05 | 91.04 | 91.11 |
| cifar100 | ls | 73.10 | 73.10 | 73.09 | 73.10 | 72.93 | 73.07 | 73.10 |
| cifar100 | re | 73.14 | 73.16 | 73.12 | 73.12 | 72.91 | 73.06 | 73.14 |
| cifar100 | reinforce | 73.13 | 73.11 | 73.10 | 73.09 | 72.95 | 73.05 | 73.11 |
| cifar100 | rs | 72.57 | 72.73 | 72.88 | 72.66 | 72.51 | 72.57 | 72.67 |

## Reductions vs size-matched random (Wilcoxon, Holm-corrected)

- ImageNet16-120|ls | no_none: Δmean=-0.036 CI[-0.107,+0.035] HL=-0.039 +/0/-=72/32/96 p_holm=0.6708 [n.s. reduction_worse]
- ImageNet16-120|ls | synflow50: Δmean=-0.036 CI[-0.111,+0.039] HL=-0.022 +/0/-=76/35/89 p_holm=0.8563 [n.s. reduction_worse]
- ImageNet16-120|ls | naswot50: Δmean=-0.080 CI[-0.153,-0.007] HL=-0.121 +/0/-=55/48/97 p_holm=0.0201 [SIG reduction_worse]
- ImageNet16-120|ls | param50: Δmean=-0.028 CI[-0.105,+0.048] HL=-0.022 +/0/-=79/35/86 p_holm=0.9282 [n.s. reduction_worse]
- ImageNet16-120|re | no_none: Δmean=+0.125 CI[+0.045,+0.204] HL=+0.133 +/0/-=98/21/81 p_holm=0.0705 [n.s. reduction_better]
- ImageNet16-120|re | synflow50: Δmean=+0.029 CI[-0.057,+0.114] HL=+0.022 +/0/-=91/29/80 p_holm=1.0000 [n.s. reduction_better]
- ImageNet16-120|re | naswot50: Δmean=+0.065 CI[-0.018,+0.149] HL=+0.078 +/0/-=82/39/79 p_holm=1.0000 [n.s. reduction_better]
- ImageNet16-120|re | param50: Δmean=+0.055 CI[-0.028,+0.139] HL=+0.039 +/0/-=85/37/78 p_holm=1.0000 [n.s. reduction_better]
- ImageNet16-120|reinforce | no_none: Δmean=+0.085 CI[-0.001,+0.168] HL=+0.103 +/0/-=86/29/85 p_holm=0.4133 [n.s. reduction_better]
- ImageNet16-120|reinforce | synflow50: Δmean=+0.104 CI[+0.023,+0.186] HL=+0.122 +/0/-=95/31/74 p_holm=0.1551 [n.s. reduction_better]
- ImageNet16-120|reinforce | naswot50: Δmean=+0.201 CI[+0.122,+0.281] HL=+0.267 +/0/-=111/33/56 p_holm=0.0000 [SIG reduction_better]
- ImageNet16-120|reinforce | param50: Δmean=+0.094 CI[+0.005,+0.183] HL=+0.122 +/0/-=98/27/75 p_holm=0.2507 [n.s. reduction_better]
- ImageNet16-120|rs | no_none: Δmean=+0.061 CI[-0.055,+0.174] HL=+0.053 +/0/-=102/4/94 p_holm=0.3783 [n.s. reduction_better]
- ImageNet16-120|rs | synflow50: Δmean=+0.199 CI[+0.087,+0.310] HL=+0.217 +/0/-=117/5/78 p_holm=0.0023 [SIG reduction_better]
- ImageNet16-120|rs | naswot50: Δmean=+0.143 CI[+0.022,+0.266] HL=+0.161 +/0/-=111/10/79 p_holm=0.0690 [n.s. reduction_better]
- ImageNet16-120|rs | param50: Δmean=+0.183 CI[+0.069,+0.298] HL=+0.178 +/0/-=105/10/85 p_holm=0.0387 [SIG reduction_better]
- cifar10-valid|ls | no_none: Δmean=-0.071 CI[-0.093,-0.048] HL=-0.111 +/0/-=30/49/121 p_holm=0.0000 [SIG reduction_worse]
- cifar10-valid|ls | synflow50: Δmean=-0.052 CI[-0.068,-0.037] HL=-0.113 +/0/-=15/95/90 p_holm=0.0000 [SIG reduction_worse]
- cifar10-valid|ls | naswot50: Δmean=-0.052 CI[-0.067,-0.036] HL=-0.113 +/0/-=14/96/90 p_holm=0.0000 [SIG reduction_worse]
- cifar10-valid|ls | param50: Δmean=-0.053 CI[-0.069,-0.037] HL=-0.113 +/0/-=16/92/92 p_holm=0.0000 [SIG reduction_worse]
- cifar10-valid|re | no_none: Δmean=-0.033 CI[-0.059,-0.007] HL=-0.068 +/0/-=50/40/110 p_holm=0.0032 [SIG reduction_worse]
- cifar10-valid|re | synflow50: Δmean=-0.020 CI[-0.038,-0.001] HL=-0.035 +/0/-=48/70/82 p_holm=0.0448 [SIG reduction_worse]
- cifar10-valid|re | naswot50: Δmean=-0.019 CI[-0.037,+0.000] HL=-0.023 +/0/-=51/74/75 p_holm=0.2489 [n.s. reduction_worse]
- cifar10-valid|re | param50: Δmean=-0.024 CI[-0.043,-0.005] HL=-0.047 +/0/-=41/74/85 p_holm=0.0047 [SIG reduction_worse]
- cifar10-valid|reinforce | no_none: Δmean=-0.034 CI[-0.058,-0.010] HL=-0.068 +/0/-=51/47/102 p_holm=0.0075 [SIG reduction_worse]
- cifar10-valid|reinforce | synflow50: Δmean=-0.014 CI[-0.036,+0.007] HL=-0.007 +/0/-=66/50/84 p_holm=1.0000 [n.s. reduction_worse]
- cifar10-valid|reinforce | naswot50: Δmean=-0.002 CI[-0.023,+0.019] HL=+0.000 +/0/-=67/57/76 p_holm=1.0000 [n.s. reduction_worse]
- cifar10-valid|reinforce | param50: Δmean=+0.001 CI[-0.029,+0.030] HL=-0.005 +/0/-=74/37/89 p_holm=1.0000 [n.s. reduction_worse]
- cifar10-valid|rs | no_none: Δmean=+0.078 CI[+0.040,+0.117] HL=+0.079 +/0/-=118/13/69 p_holm=0.0014 [SIG reduction_better]
- cifar10-valid|rs | synflow50: Δmean=+0.068 CI[+0.028,+0.109] HL=+0.073 +/0/-=113/11/76 p_holm=0.0049 [SIG reduction_better]
- cifar10-valid|rs | naswot50: Δmean=+0.031 CI[-0.011,+0.072] HL=+0.033 +/0/-=108/7/85 p_holm=0.2270 [n.s. reduction_better]
- cifar10-valid|rs | param50: Δmean=+0.064 CI[+0.021,+0.106] HL=+0.075 +/0/-=119/6/75 p_holm=0.0129 [SIG reduction_better]
- cifar100|ls | no_none: Δmean=+0.166 CI[+0.111,+0.224] HL=+0.197 +/0/-=91/47/62 p_holm=0.0019 [SIG reduction_better]
- cifar100|ls | synflow50: Δmean=+0.022 CI[-0.021,+0.066] HL=-0.073 +/0/-=34/86/80 p_holm=0.0037 [SIG reduction_worse]
- cifar100|ls | naswot50: Δmean=+0.022 CI[-0.021,+0.068] HL=-0.073 +/0/-=33/85/82 p_holm=0.0025 [SIG reduction_worse]
- cifar100|ls | param50: Δmean=+0.022 CI[-0.020,+0.068] HL=-0.073 +/0/-=34/87/79 p_holm=0.0038 [SIG reduction_worse]
- cifar100|re | no_none: Δmean=+0.205 CI[+0.146,+0.265] HL=+0.247 +/0/-=98/47/55 p_holm=0.0000 [SIG reduction_better]
- cifar100|re | synflow50: Δmean=+0.078 CI[+0.022,+0.134] HL=+0.063 +/0/-=80/53/67 p_holm=0.4538 [n.s. reduction_better]
- cifar100|re | naswot50: Δmean=+0.095 CI[+0.037,+0.153] HL=+0.137 +/0/-=74/65/61 p_holm=0.2931 [n.s. reduction_better]
- cifar100|re | param50: Δmean=+0.061 CI[+0.003,+0.121] HL=+0.047 +/0/-=75/60/65 p_holm=0.6882 [n.s. reduction_better]
- cifar100|reinforce | no_none: Δmean=+0.151 CI[+0.094,+0.210] HL=+0.183 +/0/-=82/51/67 p_holm=0.0898 [n.s. reduction_better]
- cifar100|reinforce | synflow50: Δmean=+0.065 CI[+0.010,+0.119] HL=+0.010 +/0/-=62/64/74 p_holm=1.0000 [n.s. reduction_worse]
- cifar100|reinforce | naswot50: Δmean=+0.057 CI[+0.006,+0.111] HL=+0.000 +/0/-=55/69/76 p_holm=1.0000 [n.s. reduction_worse]
- cifar100|reinforce | param50: Δmean=+0.041 CI[-0.029,+0.111] HL=+0.037 +/0/-=80/45/75 p_holm=1.0000 [n.s. reduction_better]
- cifar100|rs | no_none: Δmean=+0.373 CI[+0.265,+0.477] HL=+0.417 +/0/-=138/7/55 p_holm=0.0000 [SIG reduction_better]
- cifar100|rs | synflow50: Δmean=+0.100 CI[-0.001,+0.199] HL=+0.093 +/0/-=99/13/88 p_holm=0.2567 [n.s. reduction_better]
- cifar100|rs | naswot50: Δmean=+0.153 CI[+0.047,+0.259] HL=+0.113 +/0/-=110/5/85 p_holm=0.1479 [n.s. reduction_better]
- cifar100|rs | param50: Δmean=+0.089 CI[-0.024,+0.203] HL=+0.063 +/0/-=100/9/91 p_holm=0.2948 [n.s. reduction_better]
