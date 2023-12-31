## Simulations summary

Each weak lensing map is constructed from three simulation boxes covering different redshift ranges, with different box sizes and resolutions. The priors over which cosmological parameters are samples are:

$0.06 < \Omega_m < 0.65$ and $0.662 < S_8 < 0.966$

Fix cosmological parameters: $h=0.7$, $\Omega_b=0.046$, $n_s=0.97$ (to the same as the HSC modck simulations).

### Largest box covering $z>1$ of size 1536 Mpc/h

Stored on HPSS at: `/home/b/biwei/HSC/fastpm_box1536`
This box uses the FastPM quasi N-body code with $1536^3$ particles and 0.5 Mpc/h force resolution and 15 times steps.

### Medium box covering $0.42<z<1$ of size 704 Mpc/h

Stored on HPSS at: `/home/b/biwei/HSC/fastpm_box704`
This box uses the FastPM code with $2816^3$ particles and 0.125 Mpc/h force resolution and 60 time steps.

### Smallest box covering $z<0.42$ of size 320 Mpc/h

Stored on HPSS at: `/home/b/biwei/HSC/MP-Gadget`
This simulation uses MP-Gadget and has $960^3$ particles with 0.03 Mpc/h force resolution with adaptive time steps.

## Retrieving from HPSS, file structure, and dataset sizes

The layout of the FastPM files is as so:

```
fastpm_box1536/
├── <Cosmology index>
│   ├── Om_<Omega_m>_S8_<S_8>_<scale_factor>
│   │   ├── 1
│   │   │   ├── Position
│   │   │   ├── Velocity # (only recorded for the first step)
│   │   │   ├── ...
│   │   ├── Header
│   │   │   ├── ...
```

The output structure of the MP-Gadget simulations is explained in the documentation [here](https://www.overleaf.com/project/5cca7933041f2a71812ee0e0).

There are 101 cosmologies in total. Each particle has an associated position and velocity, and is captured at 19 timessteps so is described by 6x19=114 Float32 numbers. The total number of particles in each box differs, and so the total size in each box is:

- 1536 Mpc/h box:  $3 \times 32~{\rm Bits} \times (1536^3~{\rm particles}) * (101~{\rm cosmologies}) * (16~{\rm Snapshots}) = 83.4~{\rm TB}$
- 704 Mpc/h box: $3 \times 32~{\rm Bits} \times (2816^3~{\rm particles}) * (101~{\rm cosmologies}) * (19~{\rm Snapshots}) = 1.2~{\rm TB}$
- 320 Mpc/h box: $3 \times 32~{\rm Bits} \times (960^3~{\rm particles}) * (101~{\rm cosmologies}) = 0.1~{\rm TB}$
