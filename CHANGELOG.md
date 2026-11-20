# Changelog

## [6.1.0](https://github.com/CoreySpohn/optixstuff/compare/v6.0.0...v6.1.0) (2026-11-20)


### Features

* **instrument:** keep OpticalPath as a deprecated alias of Instrument ([3966497](https://github.com/CoreySpohn/optixstuff/commit/3966497038ec8483c4fdb7f041c5b50a2a67bc29))


### Miscellaneous Chores

* release 6.1.0 ([9875570](https://github.com/CoreySpohn/optixstuff/commit/9875570bcffad65a04eb0caed8f082d7450320ac))

## [6.0.0](https://github.com/CoreySpohn/optixstuff/compare/v5.0.0...v6.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* reduce the scalar sky and stellar leakage to annular means of the aperture integral
* mark transmission intensive and integrate native PSFs by measure
* declare spectral breaks on hardware models and bind photon and electron rates to their quadrature
* split the coronagraph contract into performance, imaging and scene capabilities with named sky responses
* add FocalResponse as the one calibrated complex speckle response
* add SampledProfile for standalone sampled curves and drop the zero-filled performance record
* align frame plans to the rates' band axis and refuse ambiguous shapes
* make Matched an inverse-variance aperture evaluated from scene rates at the exposure query
* carry separate mean and variance moments for weighted apertures and integrate hard-edged rules over one region
* bind fixed-angular apertures to quadrature nodes with prepare_aperture and the prepared and stellar_frame contract methods
* own aperture rules and performance records, add bandpass, and drop the yippy backend
* optical elements hold a hwoutils Transmission; delete SpectralThroughput
* contract maps on AbstractCoronagraph; native grid=None reads; drop noise floors and off-contract methods
* delete ExposureConfig, make dispersion_px public and read geometric centres from the grid
* detector pixel scale is a validated static host float so readers jit with the path as an argument
* one linear speckle field with the stable cross-term law, node-wavelength scaling and a day clock
* put coronagraph products on shared grids with primary diameters, a centred fold and one performance record per rule
* one detector contract with photon rates, a frame plan and a draw that matches its variance

### Features

* add FocalResponse as the one calibrated complex speckle response ([b20c708](https://github.com/CoreySpohn/optixstuff/commit/b20c708b655b94cd06ea8fc3acb26326952e7314))
* add SampledProfile for standalone sampled curves and drop the zero-filled performance record ([650f1b7](https://github.com/CoreySpohn/optixstuff/commit/650f1b787c8f97c1db1148f99fd167350fda9b06))
* bind fixed-angular apertures to quadrature nodes with prepare_aperture and the prepared and stellar_frame contract methods ([027cd1b](https://github.com/CoreySpohn/optixstuff/commit/027cd1b46b753ba3e80e2ec7a9aaf3ad3d2b569c))
* contract maps on AbstractCoronagraph; native grid=None reads; drop noise floors and off-contract methods ([816ba79](https://github.com/CoreySpohn/optixstuff/commit/816ba7901e62ba9adf489a0a7132679edaaa9778))
* **coronagraph:** accept a stellar diameter in leakage queries and add leakage_density_map ([6132543](https://github.com/CoreySpohn/optixstuff/commit/6132543bea44556aefda1c79b341a2923b2816fb))
* declare spectral breaks on hardware models and bind photon and electron rates to their quadrature ([2f34050](https://github.com/CoreySpohn/optixstuff/commit/2f340505cfb52eb140d955adf96a6fc22c325eab))
* detectors report their QE breakpoints for band quadrature ([ada25d1](https://github.com/CoreySpohn/optixstuff/commit/ada25d19f33419f0e63511fae7e028a614fa3eec))
* make Matched an inverse-variance aperture evaluated from scene rates at the exposure query ([c2bfa51](https://github.com/CoreySpohn/optixstuff/commit/c2bfa515bb34416c617eae2f530f1d197a830a10))
* one detector contract with photon rates, a frame plan and a draw that matches its variance ([e463415](https://github.com/CoreySpohn/optixstuff/commit/e463415c4bac98d63ca61a67c383339efd4c4751))
* one linear speckle field with the stable cross-term law, node-wavelength scaling and a day clock ([c2cf7cb](https://github.com/CoreySpohn/optixstuff/commit/c2cf7cb8a670d0cfcda3bec15812753338707a9d))
* own aperture rules and performance records, add bandpass, and drop the yippy backend ([15d38cb](https://github.com/CoreySpohn/optixstuff/commit/15d38cb23450bcd0b0dda106fa9d75d0e616f488))
* put coronagraph products on shared grids with primary diameters, a centred fold and one performance record per rule ([03c46a6](https://github.com/CoreySpohn/optixstuff/commit/03c46a6c991cb9b82483b161bb7450ce45c66b8e))
* split the coronagraph contract into performance, imaging and scene capabilities with named sky responses ([f36da34](https://github.com/CoreySpohn/optixstuff/commit/f36da343288629265b8017b56bea8e92889ab50a))


### Bug Fixes

* align frame plans to the rates' band axis and refuse ambiguous shapes ([0a8ad40](https://github.com/CoreySpohn/optixstuff/commit/0a8ad40150bb2c5ff863ebe913928671d964952b))
* carry separate mean and variance moments for weighted apertures and integrate hard-edged rules over one region ([4476403](https://github.com/CoreySpohn/optixstuff/commit/4476403e0d2f70cec8eaecc16425cf638e7178c5))
* **coronagraph:** resample native products on their own grid instead of returning them verbatim ([fb80cf4](https://github.com/CoreySpohn/optixstuff/commit/fb80cf44bc53b85c7f6534f6f07966b777fa4e75))
* detector pixel scale is a validated static host float so readers jit with the path as an argument ([5f94077](https://github.com/CoreySpohn/optixstuff/commit/5f940779ccdf21466dc2145e5e441572bb52ba4f))
* **optical-path:** reject a primary whose diameter disagrees with the coronagraph frame ([5ffa85c](https://github.com/CoreySpohn/optixstuff/commit/5ffa85cdb4a38a931832241074106d969edaf547))
* reduce the scalar sky and stellar leakage to annular means of the aperture integral ([5895336](https://github.com/CoreySpohn/optixstuff/commit/589533660dc799e70e33a78ba2af85af616e2b8a))


### Code Refactoring

* delete ExposureConfig, make dispersion_px public and read geometric centres from the grid ([ea44609](https://github.com/CoreySpohn/optixstuff/commit/ea44609c8be960d070b92807553a06e591948aff))
* mark transmission intensive and integrate native PSFs by measure ([ef602df](https://github.com/CoreySpohn/optixstuff/commit/ef602df71fe4bd8b57ba3e580d2f0761ddd3b0f3))
* optical elements hold a hwoutils Transmission; delete SpectralThroughput ([bb00975](https://github.com/CoreySpohn/optixstuff/commit/bb00975cd9beab205e06eda0730765f8a5ec32ec))

## [5.0.0](https://github.com/CoreySpohn/optixstuff/compare/v4.1.0...v5.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **exposure:** exposure start and speckle clock origin are days from the scene epoch

### Features

* **exposure:** exposure start and speckle clock origin are days from the scene epoch ([59ce2c8](https://github.com/CoreySpohn/optixstuff/commit/59ce2c83de150c70ff1875eb0d5a3c20d41c6430))

## [4.1.0](https://github.com/CoreySpohn/optixstuff/compare/v4.0.1...v4.1.0) (2026-11-12)


### Features

* **primary:** add SimplePrimary.from_yip ([f8f563e](https://github.com/CoreySpohn/optixstuff/commit/f8f563e94eeb326eebb5a4600236ad20b79068b3))


### Bug Fixes

* **detector:** default dqe to 1.0 so the default means no QE degradation ([735a651](https://github.com/CoreySpohn/optixstuff/commit/735a651607741091a494b8a0f710821d71da0bb5))

## [4.0.1](https://github.com/CoreySpohn/optixstuff/compare/v4.0.0...v4.0.1) (2026-11-06)


### Bug Fixes

* route yippy stellar and sky products through their own frame ([b9a029b](https://github.com/CoreySpohn/optixstuff/commit/b9a029b2b03a96564b443ad14aacfe3fdbb9b62b))

## [4.0.0](https://github.com/CoreySpohn/optixstuff/compare/v3.0.0...v4.0.0) (2026-11-05)


### ⚠ BREAKING CHANGES

* **coronagraph:** AbstractCoronagraph.core_mean_intensity implementations must accept the keyword-only stellar_diam_lod argument.
* **coronagraph:** YippyCoronagraph.core_mean_intensity and noise_floor_ayo now return values per (lambda/D)^2, the AbstractCoronagraph contract, instead of yippy's per-pixel fraction of the star's light; both grow by 1/pixel_scale_lod**2 (16x at 0.25 lambda/D pixels). Multiply either by core_area for the fraction in the aperture, and stop dividing by pixel_scale_lod**2 downstream.
* American spelling throughout, rename segment_centres_m to segment_centers_m

### Features

* **coronagraph:** add stellar_diam_lod to core_mean_intensity and noise floor ([8cd9749](https://github.com/CoreySpohn/optixstuff/commit/8cd97496c3434ea7bd4effaf2b6f2f1fb095a294))
* **speckle:** exposure-averaged realization on the speckle-field contract ([d163bf6](https://github.com/CoreySpohn/optixstuff/commit/d163bf6a0a821b1d31bed46925ef63a4c19145de))


### Bug Fixes

* American spelling throughout, rename segment_centres_m to segment_centers_m ([ccf79e1](https://github.com/CoreySpohn/optixstuff/commit/ccf79e14d8aee487d1dcabd2cf08b4a3893b2a36))
* **coronagraph:** report the yippy AYO noise floor per (lambda/D)^2 ([3add7c7](https://github.com/CoreySpohn/optixstuff/commit/3add7c7c79f8153cfafa0a7409b0ed497c001ee4))
* **coronagraph:** report yippy core mean intensity per (lambda/D)^2 ([dc55aaa](https://github.com/CoreySpohn/optixstuff/commit/dc55aaaca375bb5e45b34d93df0ff9ad5c16fbd5))

## [3.0.0](https://github.com/CoreySpohn/optixstuff/compare/v2.0.0...v3.0.0) (2026-07-20)


### ⚠ BREAKING CHANGES

* **coronagraph:** sampling-explicit image contract -- wavelength-explicit stellar_map/source_psfs/background_transmission/extended_scene, table SPI base, MultiBandCoronagraph

### Features

* **coronagraph:** sampling-explicit image contract -- wavelength-explicit stellar_map/source_psfs/background_transmission/extended_scene, table SPI base, MultiBandCoronagraph ([76c578b](https://github.com/CoreySpohn/optixstuff/commit/76c578be46ee79281dbb24ef44e3af4708fd7155))

## [2.0.0](https://github.com/CoreySpohn/optixstuff/compare/v1.5.0...v2.0.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* **disperser:** sky_pitch_arcsec replaces sky_pitch_lod -- the cube grid is fixed-angle, so the pitch is a plain angle

### Features

* **disperser:** sky_pitch_lod + psflet_pack_path descriptor fields ([d2ae639](https://github.com/CoreySpohn/optixstuff/commit/d2ae6392a5f24a2c3b43413149dd7e89b7125600))
* **primary:** exact segment geometry -- optional segment_point_to_point_m and inscribed_diameter_m, segment_pitch_m, flat-top segment_centres_m matching the HWO EAC baseline pupils ([4c5b4b7](https://github.com/CoreySpohn/optixstuff/commit/4c5b4b71843450339c2a330d74833a5b53a5431d))
* **speckle:** add AbstractSpeckleField interface and OpticalPath.speckle ([fb7b501](https://github.com/CoreySpohn/optixstuff/commit/fb7b501a2b105350786f757c7c5c761f8bfef2b0))


### Code Refactoring

* **disperser:** sky_pitch_arcsec replaces sky_pitch_lod -- the cube grid is fixed-angle, so the pitch is a plain angle ([31041e2](https://github.com/CoreySpohn/optixstuff/commit/31041e2b85402b083ce2e7913100cfc2919452f3))

## [1.5.0](https://github.com/CoreySpohn/optixstuff/compare/v1.4.0...v1.5.0) (2026-06-27)


### Features

* **detector:** wavelength-dependent QE via optional qe_curve ([bf7bc6f](https://github.com/CoreySpohn/optixstuff/commit/bf7bc6f8272ff2d011551f3b1b78243c8ada5572))
* **primary:** add SegmentedPrimary carrying hex segment geometry ([fe7cab9](https://github.com/CoreySpohn/optixstuff/commit/fe7cab9bb6153874b029ead7d2cc888ca9e8709b))

## [1.4.0](https://github.com/CoreySpohn/optixstuff/compare/v1.3.0...v1.4.0) (2026-06-23)


### Miscellaneous Chores

* release 1.4.0 ([39c1d25](https://github.com/CoreySpohn/optixstuff/commit/39c1d25741f3aaa47984d32e0e6c7917933a59ba))

## [1.3.0](https://github.com/CoreySpohn/optixstuff/compare/v1.2.0...v1.3.0) (2026-06-23)


### Features

* Add Zenodo citation ([0028643](https://github.com/CoreySpohn/optixstuff/commit/0028643e1f4079db6543a1fc43ddf353b7df226b))

## [1.2.0](https://github.com/CoreySpohn/optixstuff/compare/v1.1.0...v1.2.0) (2026-06-18)


### Features

* Add reference wavelength for the psflets ([a2fe241](https://github.com/CoreySpohn/optixstuff/commit/a2fe241ee778ef54c4832e68950fe5268f8a8b1c))
* **disperser:** wavelength-dependent throughput via composed optical element ([c902c88](https://github.com/CoreySpohn/optixstuff/commit/c902c88ae08447e5dfbc868cbcc51a254098877e))

## [1.1.0](https://github.com/CoreySpohn/optixstuff/compare/v1.0.2...v1.1.0) (2026-05-30)


### Features

* add AbstractDisperser and LensletDisperser descriptors ([6e9cd21](https://github.com/CoreySpohn/optixstuff/commit/6e9cd210c3b6f9b8af3b4b8a72cc85e412288c39))
* **detector:** add per-pixel noise_variance (shot + dark + CIC + read) ([f983b0d](https://github.com/CoreySpohn/optixstuff/commit/f983b0dc628e6a2ef8f6489a396c18ff2fff173a))
* **optical_path:** add optional disperser descriptor field ([06e486e](https://github.com/CoreySpohn/optixstuff/commit/06e486e31daaa15773d91f2b586ad7ec16359db5))

## [1.0.2](https://github.com/CoreySpohn/optixstuff/compare/v1.0.1...v1.0.2) (2026-05-25)


### Bug Fixes

* Add mermaid rendering to docs ([26e1cf0](https://github.com/CoreySpohn/optixstuff/commit/26e1cf0f8b43fe9ff16fe9c66e7bd877adf9ccd6))

## [1.0.1](https://github.com/CoreySpohn/optixstuff/compare/v1.0.0...v1.0.1) (2026-05-25)


### Bug Fixes

* Add readthedocs yaml file ([4f2380e](https://github.com/CoreySpohn/optixstuff/commit/4f2380e3e14cdecd40059bbec1a1d9d8c5684512))

## [1.0.0](https://github.com/CoreySpohn/optixstuff/compare/v0.2.0...v1.0.0) (2026-05-25)


### ⚠ BREAKING CHANGES

* Docs and refactor to standardize function calls
* SimpleDetector -> IdealDetector (clarifies that the class ignores wavelength-dependent QE and noise contributions); ConstantThroughputElement -> ConstantThroughput and LinearThroughputElement -> LinearThroughput (the Element suffix was redundant with the inheritance from AbstractUniformElement); AbstractScalarOnlyCoronagraph -> AbstractScalarCoronagraph (the Only qualifier was redundant); Exposure -> ExposureConfig (settings-struct role explicit). README ecosystem diagram replaced with Mermaid and sphinxcontrib-mermaid added to the docs build.

### Features

* Docs and refactor to standardize function calls ([5e5ab2c](https://github.com/CoreySpohn/optixstuff/commit/5e5ab2c28ac3d4dc387ba7eee7407740f95648c3))
* rename detector/throughput classes and Exposure for naming consistency ([10beac7](https://github.com/CoreySpohn/optixstuff/commit/10beac7755e38c1743aa91931739453c9a5f50ed))

## [0.2.0](https://github.com/CoreySpohn/optixstuff/compare/v0.1.0...v0.2.0) (2026-05-24)


### Features

* **optixstuff:** Add __repr__ and OpticalPath.from_default_setup ([bef51c0](https://github.com/CoreySpohn/optixstuff/commit/bef51c03f2a5845ffd1db3d613cf41d502ede0fa))
* **optixstuff:** Add readout_source_electrons_thinned for fast detector readout ([f3e7588](https://github.com/CoreySpohn/optixstuff/commit/f3e7588a1253b7cb9ebc30ec7f20f14634656527))


### Bug Fixes

* **optixstuff:** Accept yippy.EqxCoronagraph in OpticalPath.from_default_setup ([fed5b2d](https://github.com/CoreySpohn/optixstuff/commit/fed5b2d7a0e422f4a7738861ba7d60f80db188ba))
* **optixstuff:** Use yippy.fetch_yip in integration test (renamed upstream) ([8261c60](https://github.com/CoreySpohn/optixstuff/commit/8261c60be8da0506fec8f7ad7cc510c0d341287f))

## [0.1.0](https://github.com/CoreySpohn/optixstuff/compare/v0.0.1...v0.1.0) (2026-04-23)


### Features

* **detector:** split add_noise into add_source_electrons + add_noise_electrons ([6f76008](https://github.com/CoreySpohn/optixstuff/commit/6f760089d49d912c74a6c119dad174a37b3264d8))
* **exposure:** adopt Exposure module from coronagraphoto ([21eeb20](https://github.com/CoreySpohn/optixstuff/commit/21eeb20a7cffd06e48b422b3f30e8a93ebd91335))
* **optical_path:** add n_channels and npix_multiplier fields ([93a4b51](https://github.com/CoreySpohn/optixstuff/commit/93a4b5152f06adf7b7abd9a12bd7b6b1ef4ecc31))
* **yippy_coronagraph:** add create_psfs and psf_datacube for image consumers ([b74e724](https://github.com/CoreySpohn/optixstuff/commit/b74e7248ed9c312cbcdf89c12a5af21a15fe5d87))

## 0.0.1 (2026-04-13)


### Features

* implement wavelength-dependent throughput elements and YippyCoronagraph integration ([a273eac](https://github.com/CoreySpohn/optixstuff/commit/a273eac61a97bbff00aee9e147f9b2736ac5e71a))
* initial package scaffold ([909ab74](https://github.com/CoreySpohn/optixstuff/commit/909ab747971deca8c62570620b693361f55d3419))


### Miscellaneous Chores

* release 0.0.1 ([610191e](https://github.com/CoreySpohn/optixstuff/commit/610191ec673ba90d8380aa60fa2a45592cc14334))
