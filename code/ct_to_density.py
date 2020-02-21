#!/usr/bin/env python3

rho_air = 0.00120479 # http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=104
H_air = -1024
rho_t = 0.930486 # plug H = -98 into equation 21
H_t = -98 # lower bound of soft tissue H
k = (rho_t - rho_air) / (H_t - H_air)
b = -H_air * (rho_t - rho_air) / (H_t - H_air) + rho_air

#------------------------------------------------------------
#------------------------------------------------------------
def convert(ctNumber):
    density = 0.0
    #++++++++++++++++++++++++++++++++++++++++
    # mass density
    #++++++++++++++++++++++++++++++++++++++++
    if ctNumber < -1024:
        density = 0.00120479
    elif ctNumber >= -1024 and ctNumber < -98:
        density = b + k * ctNumber

    # soft tissue, equ 21
    elif ctNumber >= -98 and ctNumber <= 14:
        density = 1.018 + 0.000893 * ctNumber

    # avoid discontinuity
    elif ctNumber > 14 and ctNumber < 23:
        density = 1.03

    # soft tissue, equ 23
    elif ctNumber >= 23 and ctNumber <= 100:
        density = 1.003 + 0.001169 * ctNumber

    # skeletal tissues, equ 19
    elif ctNumber > 100 and ctNumber < 1524:
        density = 1.017 + 0.000592 * ctNumber

    elif ctNumber >= 1524:
        density = 1.017 + 0.000592 * 1524

    return density
